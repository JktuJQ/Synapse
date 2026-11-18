{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoDeepSubsumption #-}
{-# LANGUAGE NoStarIsType #-}

-- | Long short-term memory cells and time-major sequence traversal.
module Synapse.NN.Layers.Recurrent.LSTM (LSTM (..), LSTMState (..), lstmCell, lstm) where

import Control.Monad.Trans.State.Strict (State, runState, state)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, Nat, type (*))
import Synapse.NN.Activation (sigmoid)
import Synapse.NN.Layers.Recurrent.Sequence (sequenceMemory)
import Synapse.Tensor.Ops.Definition (TensorOps)
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Math (matmul)
import Synapse.Tensor.Ops.Shape (broadcast, split4)
import Synapse.Tensor.Representation (TensorElement, TensorLike)
import qualified Synapse.Tensor.Representation as T
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

instance TensorStructure (LSTM input hidden T.Tensor a) where
    type WithTensor t (LSTM input hidden T.Tensor a) = LSTM input hidden t a

-- | Hidden and cell tensors for one layer and one batch.
data LSTMState (batch :: Nat) (hidden :: Nat) (t :: TensorLike) a = LSTMState
    { lstmHiddenState :: t '[batch, hidden] a
    , lstmCellState :: t '[batch, hidden] a
    }
    deriving (Generic)

instance TensorStructure (LSTMState batch hidden T.Tensor a) where
    type WithTensor t (LSTMState batch hidden T.Tensor a) = LSTMState batch hidden t a

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
       , Num (TensorElement a)
       , Num (t '[batch, 4 * hidden] a)
       , Floating (t '[batch, hidden] a)
       , E.Ord (t '[batch, hidden] a)
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

-- | Apply one LSTM layer to @'[time, batch, input]@.
lstm
    :: forall input hidden batch time t a
     . ( KnownNat input
       , KnownNat hidden
       , KnownNat batch
       , KnownNat time
       , KnownNat (4 * hidden)
       , TensorOps t a
       , Floating a
       , Floating (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => LSTM input hidden t a
    -> t '[time, batch, input] a
    -> State (LSTMState batch hidden t a) (t '[time, batch, hidden] a)
lstm parameters inputs = state $ \(LSTMState hidden memory) ->
    let weights =
            ( (lstmInputWeights parameters, lstmHiddenWeights parameters)
            , (lstmInputBias parameters, lstmHiddenBias parameters)
            )
        (outputs, (finalHidden, finalMemory)) =
            runState
                ( sequenceMemory
                    ( \((w, r), (b, c)) x -> state $ \(h, m) ->
                        let (output, LSTMState nextHidden nextMemory) = runState (lstmCell (LSTM w r b c) x) $ LSTMState h m
                         in (output, (nextHidden, nextMemory))
                    )
                    weights
                    inputs
                )
                (hidden, memory)
     in (outputs, LSTMState finalHidden finalMemory)
