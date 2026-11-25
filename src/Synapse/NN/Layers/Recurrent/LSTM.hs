{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Long short-term memory cells and time-major sequence traversal.
module Synapse.NN.Layers.Recurrent.LSTM (LSTM (..), LSTMState (..), lstmCell, lstm) where

import Control.Monad.Trans.State.Strict (State, state)
import Data.Array.Accelerate qualified as A
import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, Nat, natVal, type (*))
import Synapse.Autograd.Tensor.Ops.Pullback.Shape (joinMatrixChunks)
import Synapse.NN.Activation (sigmoid)
import Synapse.NN.Layers.Recurrent.Sequence (item, projectSequence, projectSequenceBackward, writeItem)
import Synapse.Tensor.Ops.Constraints (FloatingDType, OrdDType)
import Synapse.Tensor.Ops.Definition (TensorOps (..))
import Synapse.Tensor.Ops.Execution (force)
import Synapse.Tensor.Ops.Logic qualified as E
import Synapse.Tensor.Ops.Math (matmul, matmult, tmatmul)
import Synapse.Tensor.Ops.Raw.Construction (full)
import Synapse.Tensor.Ops.Reduction (sumAxes)
import Synapse.Tensor.Ops.Shape (broadcast, split4)
import Synapse.Tensor.Representation (Tensor (..), TensorElement (..), TensorLike)
import Synapse.Tensor.Structure (TensorStructure, WithTensor)

{- | Packed input, forget, candidate and output projections, in that order.

Input weights have shape @'[input, 4 * hidden]@ and recurrent weights have
shape @'[hidden, 4 * hidden]@. Each bias has shape @'[4 * hidden]@.
-}
data LSTM (input :: Nat) (hidden :: Nat) (t :: TensorLike) a = LSTM
    { lstmInputWeights :: t '[input, 4 * hidden] a
    , lstmHiddenWeights :: t '[hidden, 4 * hidden] a
    , lstmInputBias :: t '[4 * hidden] a
    , lstmHiddenBias :: t '[4 * hidden] a
    }
    deriving (Generic)

instance TensorStructure (LSTM input hidden Tensor a) where
    type WithTensor t (LSTM input hidden Tensor a) = LSTM input hidden t a

-- | Hidden and cell tensors for one layer and one batch.
data LSTMState (batch :: Nat) (hidden :: Nat) (t :: TensorLike) a = LSTMState
    { lstmHiddenState :: t '[batch, hidden] a
    , lstmCellState :: t '[batch, hidden] a
    }
    deriving (Generic)

instance TensorStructure (LSTMState batch hidden Tensor a) where
    type WithTensor t (LSTMState batch hidden Tensor a) = LSTMState batch hidden t a

{- | Apply one LSTM step and update both state tensors.

Split the affine projection into gates @(i, f, g, o)@, then compute:

@
c' = sigmoid f * c + sigmoid i * tanh g
h' = sigmoid o * tanh c'
@
-}
lstmCell
    :: forall input hidden batch t a
     . ( KnownNat input
       , KnownNat hidden
       , KnownNat batch
       , KnownNat (4 * hidden)
       , TensorOps t a
       , FloatingDType a
       , OrdDType a
       , Num (t '[batch, 4 * hidden] a)
       , Floating (t '[batch, hidden] a)
       )
    => LSTM input hidden t a
    -> t '[batch, input] a
    -> State (LSTMState batch hidden t a) (t '[batch, hidden] a)
lstmCell parameters input = state $ \(LSTMState previous memory) ->
    let gates =
            matmul input (lstmInputWeights parameters)
                + broadcast (lstmInputBias parameters)
                + matmul previous (lstmHiddenWeights parameters)
                + broadcast (lstmHiddenBias parameters)
        (i, f, g, o) = split4 gates
        nextMemory = sigmoid f * memory + sigmoid i * tanh g
        nextHidden = sigmoid o * tanh nextMemory
     in (nextHidden, LSTMState nextHidden nextMemory)

-- | Parameters, input sequence, and initial recurrent state.
type Arguments time batch input hidden a =
    (LSTM input hidden Tensor a, (Tensor '[time, batch, input] a, LSTMState batch hidden Tensor a))

-- | Hidden outputs and the final recurrent state.
type Result time batch hidden a = (Tensor '[time, batch, hidden] a, LSTMState batch hidden Tensor a)

-- | Values retained to differentiate the sequence.
data Cache time batch input hidden a = Cache
    { arguments :: Arguments time batch input hidden a
    , hiddenHistory :: Tensor '[time, batch, hidden] a
    , cellHistory :: Tensor '[time, batch, hidden] a
    , inputProjection :: Tensor '[time, batch, 4 * hidden] a
    }

-- | Evaluate sigmoid with a bounded exponential.
sigmoidElement
    :: (FloatingDType a, OrdDType a)
    => TensorElement a
    -> TensorElement a
sigmoidElement x =
    let e = exp (negate $ abs x)
        yes = unTensorElement $ recip (1 + e)
        no = unTensorElement $ e / (1 + e)
     in TensorElement $ A.cond (unTensorElement $ x E.>= 0) yes no

-- | Apply tanh to candidate gates and sigmoid to the remaining gates.
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
         in A.cond isCandidate (unTensorElement $ tanh x) (unTensorElement $ sigmoidElement x)

-- | Project the input and scan over the sequence.
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
    -> Arguments time batch input hidden a
    -> (Result time batch hidden a, Cache time batch input hidden a)
forward retainCellHistory args@(LSTM w r b c, (xs, LSTMState initialHidden initialCell)) =
    let projected = projectSequence (w, b, xs)
        condition (A.T3 i _ _) =
            A.unit $ A.the i A.< A.constant (fromIntegral $ natVal $ Proxy @time)
        body (A.T3 idx (A.T2 hidden memory) (A.T2 hs ms)) =
            let i = A.the idx
                h = Tensor @'[batch, hidden] hidden
                gates = activate @batch @hidden $ item @'[batch, 4 * hidden] i projected + matmul h r + broadcast c
                (ig, fg, gg, og) = split4 gates
                nextM = fg * Tensor memory + ig * gg
                nextH = og * tanh nextM
                Tensor hs' = writeItem i nextH (Tensor @'[time, batch, hidden] hs)
                Tensor ms' =
                    if retainCellHistory then writeItem i nextM (Tensor @'[time, batch, hidden] ms) else Tensor ms
             in A.T3 (A.map (+ 1) idx) (A.T2 (unTensor nextH) (unTensor nextM)) (A.T2 hs' ms')
        Tensor emptyH = full @'[time, batch, hidden] 0
        emptyM =
            if retainCellHistory
                then unTensor $ full @'[time, batch, hidden] @a 1
                else unTensor $ full @'[0, 0, 0] @a 0
        A.T3 _ (A.T2 finalHiddenState finalCellState) (A.T2 finalHidden finalCells) =
            A.awhile condition body $
                A.T3
                    (A.unit $ A.constant (0 :: Int))
                    (A.T2 (unTensor initialHidden) (unTensor initialCell))
                    (A.T2 emptyH emptyM)
        outputs = Tensor finalHidden
        final = LSTMState (Tensor finalHiddenState) (Tensor finalCellState)
     in ((outputs, final), Cache args outputs (Tensor finalCells) projected)

-- | Propagate output seeds backwards through the recurrence.
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
    -> Result time batch hidden a
    -> Arguments time batch input hidden a
backward cache (seeds, LSTMState lastHiddenSeed lastCellSeed) =
    let (LSTM w r b c, (xs, LSTMState h0 m0)) = arguments cache
        projected = inputProjection cache
        condition (A.T4 i _ _ _) = A.unit $ A.the i A.> 0
        body (A.T4 idx (A.T2 dhNext dcNext) (A.T2 dr0 db0) projectionSeeds) =
            let i = A.the idx - 1
                prevH =
                    Tensor @'[batch, hidden] $
                        A.acond (i A.== 0) (unTensor h0) (unTensor $ item @'[batch, hidden] (i - 1) $ hiddenHistory cache)
                prevM =
                    Tensor @'[batch, hidden] $
                        A.acond (i A.== 0) (unTensor m0) (unTensor $ item @'[batch, hidden] (i - 1) $ cellHistory cache)
                gateInput = item @'[batch, 4 * hidden] i projected + matmul prevH r + broadcast c
                (ig, fg, gg, og) = split4 (activate @batch @hidden gateInput)
                currentM = item @'[batch, hidden] i (cellHistory cache)
                dh = Tensor dhNext + item @'[batch, hidden] i seeds
                th = tanh currentM
                dc = Tensor dcNext + dh * og * (1 - th * th)
                di = dc * gg * ig * (1 - ig)
                df = dc * prevM * fg * (1 - fg)
                dg = dc * ig * (1 - gg * gg)
                dout = dh * th * og * (1 - og)
                gateGradient = force $ joinMatrixChunks @4 @hidden [di, df, dg, dout]
                nextM = dc * fg
                nextH = matmult gateGradient r
                dr = Tensor dr0 + tmatmul prevH gateGradient
                db = Tensor db0 + sumAxes @'[0] gateGradient
                Tensor nextD = writeItem i gateGradient (Tensor @'[time, batch, 4 * hidden] projectionSeeds)
             in A.T4
                    (A.map (subtract 1) idx)
                    (A.T2 (unTensor nextH) (unTensor nextM))
                    (A.T2 (unTensor dr) (unTensor db))
                    nextD
        Tensor emptyR = full @'[hidden, 4 * hidden] 0
        Tensor emptyB = full @'[4 * hidden] 0
        Tensor emptyD = full @'[time, batch, 4 * hidden] 0
        A.T4 _ (A.T2 firstHiddenSeed firstCellSeed) (A.T2 recurrentWeightSeed recurrentBiasSeed) dprojs =
            A.awhile condition body $
                A.T4
                    (A.unit $ A.constant (fromIntegral $ natVal $ Proxy @time))
                    (A.T2 (unTensor lastHiddenSeed) (unTensor lastCellSeed))
                    (A.T2 emptyR emptyB)
                    emptyD
        (dw, inputBiasSeed, dx) = projectSequenceBackward (w, b, xs) (Tensor @'[time, batch, 4 * hidden] dprojs)
        weightSeeds = LSTM dw (Tensor recurrentWeightSeed) inputBiasSeed (Tensor recurrentBiasSeed)
     in (weightSeeds, (dx, LSTMState (Tensor firstHiddenSeed) (Tensor firstCellSeed)))

-- | Apply one LSTM layer to @'[time, batch, input]@.
lstm
    :: forall input hidden batch time t a
     . ( KnownNat input
       , KnownNat hidden
       , KnownNat batch
       , KnownNat time
       , KnownNat (4 * hidden)
       , TensorOps t a
       , FloatingDType a
       , OrdDType a
       )
    => LSTM input hidden t a
    -> t '[time, batch, input] a
    -> State (LSTMState batch hidden t a) (t '[time, batch, hidden] a)
lstm parameters inputs = state $ \initial ->
    customVJP
        (Proxy @(() :: Constraint))
        (fst . forward @time @batch @input @hidden False)
        (forward @time @batch @input @hidden True)
        (backward @time @batch @input @hidden)
        (parameters, (inputs, initial))
