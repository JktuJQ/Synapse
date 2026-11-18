{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Sequence traversal shared by RNN, GRU and LSTM.
module Synapse.NN.Layers.Recurrent.Sequence (sequenceMemory, sequenceHidden) where

import Control.Monad.Trans.State.Strict (State, runState, state)
import qualified Data.Array.Accelerate as A
import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat, natVal)
import qualified Synapse.Autograd.Execution as AG
import Synapse.Tensor.Ops.Definition (TensorOps (..))
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Math ()
import Synapse.Tensor.Ops.Raw.Construction (full)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement, TensorLike)
import Synapse.Tensor.Shapes.Backend (backendShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)

-- The three layers share two projection matrices and two biases.
type Weights (t :: TensorLike) input hidden gates a =
    ((t '[input, gates] a, t '[hidden, gates] a), (t '[gates] a, t '[gates] a))

type Memory (t :: TensorLike) batch hidden memory a = (t '[batch, hidden] a, t '[batch, memory] a)

type Arguments t time batch input hidden gates memory a =
    (Weights t input hidden gates a, (t '[time, batch, input] a, Memory t batch hidden memory a))

type Result t time batch hidden memory a =
    (t '[time, batch, hidden] a, Memory t batch hidden memory a)

type History a = (A.Matrix a, A.Matrix a)

packState
    :: DType a
    => Memory Tensor batch hidden memory a
    -> A.Acc (A.Matrix a, A.Matrix a)
packState (Tensor hidden, Tensor memory) = A.lift (hidden, memory)

unpackState
    :: DType a
    => A.Acc (A.Matrix a, A.Matrix a)
    -> Memory Tensor batch hidden memory a
unpackState value = let (hidden, memory) = A.unlift value in (Tensor hidden, Tensor memory)

packWeights
    :: DType a
    => Weights Tensor input hidden gates a
    -> A.Acc ((A.Matrix a, A.Matrix a), (A.Vector a, A.Vector a))
packWeights ((Tensor w, Tensor r), (Tensor b, Tensor c)) =
    A.lift ((w, r), (b, c))

unpackWeights
    :: forall input hidden gates a
     . DType a
    => A.Acc ((A.Matrix a, A.Matrix a), (A.Vector a, A.Vector a))
    -> Weights Tensor input hidden gates a
unpackWeights value =
    let (matrices, biases) =
            A.unlift value :: (A.Acc (A.Matrix a, A.Matrix a), A.Acc (A.Vector a, A.Vector a))
        (w, r) = A.unlift matrices
        (b, c) = A.unlift biases
     in ((Tensor w, Tensor r), (Tensor b, Tensor c))

-- Select or replace one time step. Output ranges of different steps are disjoint.
item
    :: forall sh sequence a
     . (Shape sh, Shape sequence, DType a)
    => A.Exp Int
    -> Tensor sequence a
    -> Tensor sh a
item i (Tensor input) = Tensor $
    A.reshape (A.constant $ backendShape $ Proxy @sh) $
        A.generate (A.constant $ A.Z A.:. shapeSize @sh) $ \ix ->
            A.flatten input A.! A.index1 (i * A.constant (shapeSize @sh) + A.indexHead ix)

writeItem
    :: forall sh sequence a
     . (Shape sh, Shape sequence, DType a)
    => A.Exp Int
    -> Tensor sh a
    -> Tensor sequence a
    -> Tensor sequence a
writeItem i (Tensor value) (Tensor target) =
    Tensor
        $ A.reshape (A.shape target)
        $ A.permute
            const
            (A.flatten target)
            (\ix -> A.Just_ $ A.index1 $ i * A.constant (shapeSize @sh) + A.indexHead ix)
        $ A.flatten value

saveHistory
    :: (Shape sh, DType a)
    => A.Exp Int
    -> Tensor sh a
    -> A.Acc (A.Matrix a)
    -> A.Acc (A.Matrix a)
saveHistory i (Tensor value) history =
    A.permute const history (A.Just_ . A.index2 i . A.indexHead) $ A.flatten value

readHistory
    :: forall sh a
     . (Shape sh, DType a)
    => A.Exp Int
    -> A.Acc (A.Matrix a)
    -> Tensor sh a
readHistory i history = Tensor $
    A.reshape (A.constant $ backendShape $ Proxy @sh) $
        A.generate (A.constant $ A.Z A.:. shapeSize @sh) $
            \ix -> history A.! A.index2 i (A.indexHead ix)

-- Only differentiated execution saves states; ordinary execution uses empty histories.
traverseSequence
    :: forall time batch input hidden gates memory a
     . ( KnownNat time
       , KnownNat batch
       , KnownNat input
       , KnownNat hidden
       , KnownNat gates
       , KnownNat memory
       , DType a
       , Num a
       , Num (TensorElement a)
       )
    => Bool
    -> ( Weights Tensor input hidden gates a
         -> Tensor '[batch, input] a
         -> Memory Tensor batch hidden memory a
         -> (Tensor '[batch, hidden] a, Memory Tensor batch hidden memory a)
       )
    -> Arguments Tensor time batch input hidden gates memory a
    -> (Result Tensor time batch hidden memory a, A.Acc (History a))
traverseSequence retainHistory step (parameters, (inputs, initial)) =
    let unliftLoop value =
            A.unlift value
                :: ( A.Acc (A.Scalar Int)
                   , A.Acc (A.Matrix a, A.Matrix a)
                   , A.Acc (A.Array A.DIM3 a)
                   , A.Acc (History a)
                   )
        count = fromIntegral $ natVal $ Proxy @time
        historyLength = if retainHistory then count else 0
        emptyHistory width = A.fill (A.constant $ A.Z A.:. historyLength A.:. width) (A.constant 0)
        condition value = let (i, _, _, _) = unliftLoop value in A.unit $ A.the i A.< A.constant count
        body value =
            let (index, current, outputs, history) = unliftLoop value
                i = A.the index
                (output, next@(h, c)) = step parameters (item i inputs) $ unpackState current
                Tensor nextOutputs = writeItem i output (Tensor outputs :: Tensor '[time, batch, hidden] a)
                (hs, cs) = A.unlift history
                nextHistory =
                    if retainHistory
                        then A.lift (saveHistory i h hs, saveHistory i c cs)
                        else history
             in A.lift (A.map (+ 1) index, packState next, nextOutputs, nextHistory)
        (_, final, result, finalHistory) =
            unliftLoop $
                A.awhile condition body $
                    A.lift
                        ( A.unit $ A.constant (0 :: Int)
                        , packState initial
                        , unTensor $ full @'[time, batch, hidden] 0
                        , A.lift
                            ( emptyHistory $ shapeSize @'[batch, hidden]
                            , emptyHistory $ shapeSize @'[batch, memory]
                            )
                            :: A.Acc (History a)
                        )
     in ((Tensor result, unpackState final), finalHistory)

-- Recompute each cell's intermediates from its saved input state.
backward
    :: forall time batch input hidden gates memory a
     . ( KnownNat time
       , KnownNat batch
       , KnownNat input
       , KnownNat hidden
       , KnownNat gates
       , KnownNat memory
       , DType a
       , Num a
       , Num (TensorElement a)
       )
    => ( ( Weights Tensor input hidden gates a
         , (Tensor '[batch, input] a, Memory Tensor batch hidden memory a)
         )
         -> (Tensor '[batch, hidden] a, Memory Tensor batch hidden memory a)
         -> ( Weights Tensor input hidden gates a
            , (Tensor '[batch, input] a, Memory Tensor batch hidden memory a)
            )
       )
    -> (Arguments Tensor time batch input hidden gates memory a, A.Acc (History a))
    -> Result Tensor time batch hidden memory a
    -> Arguments Tensor time batch input hidden gates memory a
backward step ((parameters, (inputs, initial)), history) (seeds, finalSeed) =
    let unliftLoop value =
            A.unlift value
                :: ( A.Acc (A.Scalar Int)
                   , A.Acc (A.Matrix a, A.Matrix a)
                   , A.Acc ((A.Matrix a, A.Matrix a), (A.Vector a, A.Vector a))
                   , A.Acc (A.Array A.DIM3 a)
                   )
        count = fromIntegral $ natVal $ Proxy @time
        condition value = let (i, _, _, _) = unliftLoop value in A.unit $ A.the i A.> 0
        body value =
            let (index, stateSeed, parameterSeed, inputSeeds) = unliftLoop value
                i = A.the index - 1
                (hs, cs) = A.unlift history
                previous =
                    A.acond (i A.== 0) (packState initial) $
                        packState
                            ( readHistory @'[batch, hidden] (i - 1) hs
                            , readHistory @'[batch, memory] (i - 1) cs
                            )
                (((dw, dr), (db, dc)), (dx, ds)) =
                    step (parameters, (item i inputs, unpackState previous)) (item i seeds, unpackState stateSeed)
                ((w, r), (b, c)) = unpackWeights parameterSeed
                Tensor nextInputs = writeItem i dx (Tensor inputSeeds :: Tensor '[time, batch, input] a)
             in A.lift
                    ( A.map (subtract 1) index
                    , packState ds
                    , packWeights ((w + dw, r + dr), (b + db, c + dc))
                    , nextInputs
                    )
        (_, stateGradient, parameterGradient, inputGradient) =
            unliftLoop $
                A.awhile condition body $
                    A.lift
                        ( A.unit $ A.constant count
                        , packState finalSeed
                        , packWeights
                            ( (full @'[input, gates] 0, full @'[hidden, gates] 0)
                            , (full @'[gates] 0, full @'[gates] 0)
                            )
                        , unTensor $ full @'[time, batch, input] 0
                        )
     in (unpackWeights parameterGradient, (Tensor inputGradient, unpackState stateGradient))

sequenceMemory
    :: forall time batch input hidden gates memory t a
     . ( KnownNat time
       , KnownNat batch
       , KnownNat input
       , KnownNat hidden
       , KnownNat gates
       , KnownNat memory
       , TensorOps t a
       , Floating a
       , Floating (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => ( forall u
          . ( TensorOps u a
            , forall sh. Shape sh => Floating (u sh a)
            , forall sh. Shape sh => E.Ord (u sh a)
            )
         => Weights u input hidden gates a
         -> u '[batch, input] a
         -> State (Memory u batch hidden memory a) (u '[batch, hidden] a)
       )
    -> Weights t input hidden gates a
    -> t '[time, batch, input] a
    -> State (Memory t batch hidden memory a) (t '[time, batch, hidden] a)
sequenceMemory step parameters inputs = state $ \initial ->
    customVJP
        (Proxy @(() :: Constraint))
        (fst . traverseSequence @time False stepForward)
        ( \arguments ->
            let (result, history) = traverseSequence @time True stepForward arguments
             in (result, (arguments, history))
        )
        (backward @time stepBackward)
        (parameters, (inputs, initial))
  where
    stepForward p x = runState $ step p x
    stepBackward arguments = snd $ AG.pullback (\(p, (x, s)) -> runState (step p x) s) arguments

-- RNN and GRU have no cell memory; the second tensor contains no elements.
sequenceHidden
    :: forall time batch input hidden gates t a
     . ( KnownNat time
       , KnownNat batch
       , KnownNat input
       , KnownNat hidden
       , KnownNat gates
       , TensorOps t a
       , Floating a
       , Floating (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => ( forall u
          . ( TensorOps u a
            , forall sh. Shape sh => Floating (u sh a)
            , forall sh. Shape sh => E.Ord (u sh a)
            )
         => Weights u input hidden gates a
         -> u '[batch, input] a
         -> State (u '[batch, hidden] a) (u '[batch, hidden] a)
       )
    -> Weights t input hidden gates a
    -> t '[time, batch, input] a
    -> State (t '[batch, hidden] a) (t '[time, batch, hidden] a)
sequenceHidden step parameters inputs = state $ \initial ->
    let (outputs, (final, _)) =
            runState
                ( sequenceMemory
                    ( \p x -> state $ \(h, memory) ->
                        let (output, next) = runState (step p x) h
                         in (output, (next, memory))
                    )
                    parameters
                    inputs
                )
                (initial, fromTensor (full @'[batch, 0] 0))
     in (outputs, final)
