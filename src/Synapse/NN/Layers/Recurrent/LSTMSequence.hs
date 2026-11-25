{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}

-- | Sequence evaluation with a shared input projection and an explicit LSTM pullback.
module Synapse.NN.Layers.Recurrent.LSTMSequence (sequenceLSTM) where

import Control.Monad.Trans.State.Strict (State, state)
import Data.Array.Accelerate qualified as A
import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat, natVal, type (*))
import Synapse.Autograd.Execution qualified as AG
import Synapse.Autograd.Tensor.Ops.Pullback.Shape (joinMatrixChunks)
import Synapse.NN.Layers.Recurrent.Sequence (item, packState, unpackState, writeItem)
import Synapse.Tensor.Ops.Constraints (FloatingDType, OrdDType)
import Synapse.Tensor.Ops.Definition (TensorOps (..))
import Synapse.Tensor.Ops.Execution (force)
import Synapse.Tensor.Ops.Logic qualified as E
import Synapse.Tensor.Ops.Math (matmul)
import Synapse.Tensor.Ops.Raw.Construction (full)
import Synapse.Tensor.Ops.Reduction (sumAxes)
import Synapse.Tensor.Ops.Shape (broadcast, split4, transpose)
import Synapse.Tensor.Representation (Tensor (..), TensorElement (..), TensorLike)

type Weights (t :: TensorLike) input hidden a =
    ((t '[input, 4 * hidden] a, t '[hidden, 4 * hidden] a), (t '[4 * hidden] a, t '[4 * hidden] a))
type Arguments t time batch input hidden a =
    (Weights t input hidden a, (t '[time, batch, input] a, (t '[batch, hidden] a, t '[batch, hidden] a)))
type Result t time batch hidden a = (t '[time, batch, hidden] a, (t '[batch, hidden] a, t '[batch, hidden] a))
data Cache time batch input hidden a = Cache
    { arguments :: Arguments Tensor time batch input hidden a
    , hiddenHistory :: Tensor '[time, batch, hidden] a
    , cellHistory :: Tensor '[time, batch, hidden] a
    , inputProjection :: Tensor '[time, batch, 4 * hidden] a
    }

sigmoidFast
    :: (FloatingDType a, OrdDType a)
    => TensorElement a
    -> TensorElement a
sigmoidFast x =
    let e = exp (negate $ abs x)
        yes = unTensorElement $ recip (1 + e)
        no = unTensorElement $ e / (1 + e)
     in TensorElement $ A.cond (unTensorElement $ x E.>= 0) yes no

activate
    :: forall batch hidden a
     . (KnownNat batch, KnownNat hidden, KnownNat (4 * hidden), FloatingDType a, OrdDType a)
    => Tensor '[batch, 4 * hidden] a
    -> Tensor '[batch, 4 * hidden] a
activate value@(Tensor gates)
    | natVal (Proxy @hidden) == 0 = value
    | otherwise = force $ Tensor $ A.generate (A.shape gates) $ \ix ->
        let x = TensorElement (gates A.! ix)
            isCandidate = A.indexHead ix `A.quot` A.constant (fromIntegral $ natVal $ Proxy @hidden) A.== 2
         in unTensorElement $ ifScalar isCandidate (tanh x) (sigmoidFast x)
  where
    ifScalar p (TensorElement x) (TensorElement y) = TensorElement (A.cond p x y)

forward
    :: forall time batch input hidden a
     . ( KnownNat time
       , KnownNat batch
       , KnownNat input
       , KnownNat hidden
       , KnownNat (4 * hidden)
       , FloatingDType a
       , OrdDType a
       )
    => Bool
    -> Arguments Tensor time batch input hidden a
    -> (Result Tensor time batch hidden a, Cache time batch input hidden a)
forward retainCellHistory args@(((w, r), (b, c)), (xs, initial)) =
    let projected = matmul xs w + broadcast b
        loopValue value =
            A.unlift value
                :: (A.Acc (A.Scalar Int), A.Acc (A.Matrix a, A.Matrix a), A.Acc (A.Array A.DIM3 a, A.Array A.DIM3 a))
        condition value =
            let (i, _, _) = loopValue value
             in A.unit $ A.the i A.< A.constant (fromIntegral $ natVal $ Proxy @time)
        body value =
            let (idx, current, history) = loopValue value
                i = A.the idx
                (h, m) = unpackState @batch @hidden @hidden current
                gates = activate @batch @hidden $ item @'[batch, 4 * hidden] i projected + matmul h r + broadcast c
                (ig, fg, gg, og) = split4 gates
                nextM = fg * m + ig * gg
                nextH = og * tanh nextM
                (hs, ms) = A.unlift history
                Tensor hs' = writeItem i nextH (Tensor hs :: Tensor '[time, batch, hidden] a)
                Tensor ms' =
                    if retainCellHistory then writeItem i nextM (Tensor ms :: Tensor '[time, batch, hidden] a) else Tensor ms
             in A.lift
                    (A.map (+ 1) idx, packState (nextH, nextM), A.lift (hs', ms') :: A.Acc (A.Array A.DIM3 a, A.Array A.DIM3 a))
        Tensor emptyH = full @'[time, batch, hidden] 0
        -- Use distinct buffers for the two histories because permute can update them in place.
        emptyM =
            if retainCellHistory
                then unTensor $ full @'[time, batch, hidden] @a 1
                else unTensor $ full @'[0, 0, 0] @a 0
        (_, lastState, finalHistory) =
            loopValue $
                A.awhile condition body $
                    A.lift
                        ( A.unit (A.constant (0 :: Int))
                        , packState initial
                        , A.lift (emptyH, emptyM) :: A.Acc (A.Array A.DIM3 a, A.Array A.DIM3 a)
                        )
        (finalHidden, finalCells) = A.unlift finalHistory
        outputs = Tensor finalHidden
     in ((outputs, unpackState lastState), Cache args outputs (Tensor finalCells) projected)

backward
    :: forall time batch input hidden a
     . ( KnownNat time
       , KnownNat batch
       , KnownNat input
       , KnownNat hidden
       , KnownNat (4 * hidden)
       , FloatingDType a
       , OrdDType a
       )
    => Cache time batch input hidden a
    -> Result Tensor time batch hidden a
    -> Arguments Tensor time batch input hidden a
backward cache (seeds, lastSeeds) =
    let (((w, r), (b, c)), (xs, (h0, m0))) = arguments cache
        projected = inputProjection cache
        loopValue value =
            A.unlift value
                :: ( A.Acc (A.Scalar Int)
                   , A.Acc (A.Matrix a, A.Matrix a)
                   , A.Acc (A.Matrix a, A.Vector a)
                   , A.Acc (A.Array A.DIM3 a)
                   )
        condition value = let (i, _, _, _) = loopValue value in A.unit $ A.the i A.> 0
        body value =
            let (idx, nextSeed, parameterSeed, projectionSeeds) = loopValue value
                i = A.the idx - 1
                (dhNext, dcNext) = unpackState nextSeed
                (dr0, db0) = A.unlift parameterSeed
                prevH =
                    Tensor @'[batch, hidden] $
                        A.acond (i A.== 0) (unTensor h0) (unTensor $ item @'[batch, hidden] (i - 1) $ hiddenHistory cache)
                prevM =
                    Tensor @'[batch, hidden] $
                        A.acond (i A.== 0) (unTensor m0) (unTensor $ item @'[batch, hidden] (i - 1) $ cellHistory cache)
                gateInput = item @'[batch, 4 * hidden] i projected + matmul prevH r + broadcast c
                (ig, fg, gg, og) = split4 (activate @batch @hidden gateInput)
                currentM = item @'[batch, hidden] i (cellHistory cache)
                dh = dhNext + item @'[batch, hidden] i seeds
                th = tanh currentM
                dc = dcNext + dh * og * (1 - th * th)
                di = dc * gg * ig * (1 - ig)
                df = dc * prevM * fg * (1 - fg)
                dg = dc * ig * (1 - gg * gg)
                dout = dh * th * og * (1 - og)
                gatesT = force $ joinMatrixChunks @4 @hidden [di, df, dg, dout]
                nextH = matmul gatesT (transpose r)
                nextM = dc * fg
                dr = (Tensor dr0 :: Tensor '[hidden, 4 * hidden] a) + matmul (transpose prevH) gatesT
                db = (Tensor db0 :: Tensor '[4 * hidden] a) + sumAxes @'[0] gatesT
                Tensor nextD = writeItem i gatesT (Tensor @'[time, batch, 4 * hidden] projectionSeeds)
             in A.lift
                    ( A.map (subtract 1) idx
                    , packState (nextH, nextM)
                    , A.lift (unTensor dr, unTensor db) :: A.Acc (A.Matrix a, A.Vector a)
                    , nextD
                    )
        Tensor emptyR = full @'[hidden, 4 * hidden] 0
        Tensor emptyB = full @'[4 * hidden] 0
        Tensor emptyD = full @'[time, batch, 4 * hidden] 0
        (_, firstSeeds, parameters, dprojs) =
            loopValue $
                A.awhile condition body $
                    A.lift
                        ( A.unit (A.constant (fromIntegral $ natVal $ Proxy @time))
                        , packState lastSeeds
                        , A.lift (emptyR, emptyB) :: A.Acc (A.Matrix a, A.Vector a)
                        , emptyD
                        )
        (recurrentWeightSeed, recurrentBiasSeed) = A.unlift parameters
        (_, projectionBackward) = AG.pullback (\(pw, pb, input) -> matmul input pw + broadcast pb) (w, b, xs)
        (dw, inputBiasSeed, dx) = projectionBackward (Tensor @'[time, batch, 4 * hidden] dprojs)
        weightSeeds = ((dw, Tensor recurrentWeightSeed), (inputBiasSeed, Tensor recurrentBiasSeed))
     in (weightSeeds, (dx, unpackState firstSeeds))

sequenceLSTM
    :: forall time batch input hidden t a
     . ( KnownNat time
       , KnownNat batch
       , KnownNat input
       , KnownNat hidden
       , KnownNat (4 * hidden)
       , TensorOps t a
       , FloatingDType a
       , OrdDType a
       )
    => Weights t input hidden a
    -> t '[time, batch, input] a
    -> State (t '[batch, hidden] a, t '[batch, hidden] a) (t '[time, batch, hidden] a)
sequenceLSTM weights xs = state $ \initial ->
    customVJP
        (Proxy @(() :: Constraint))
        (fst . forward @time @batch @input @hidden False)
        (forward @time @batch @input @hidden True)
        (backward @time @batch @input @hidden)
        (weights, (xs, initial))
