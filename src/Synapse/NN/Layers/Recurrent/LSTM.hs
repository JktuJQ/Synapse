{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | Long short-term memory cells and time-major sequence traversal.
module Synapse.NN.Layers.Recurrent.LSTM (LSTM (..), LSTMState (..), lstmCell, lstm) where

import Control.Monad.Trans.State.Strict (State, state)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, Nat, type (*))
import Synapse.NN.Activation (sigmoid)
import Synapse.NN.Layers.Recurrent.Gates (split4)
import Synapse.Tensor.Ops (TensorOps)
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Math (matmul)
import Synapse.Tensor.Ops.Sequence (mapAccum)
import Synapse.Tensor.Ops.Shape (broadcast)
import Synapse.Tensor.Type (TensorElement, TensorLike, WithTensor)
import qualified Synapse.Tensor.Type as T

-- | Packed input, forget, candidate and output projections, in that order.
-- Input weights have shape @'[input, 4 * hidden]@ and recurrent weights have
-- shape @'[hidden, 4 * hidden]@. Each bias has shape @'[4 * hidden]@.
data LSTM (input :: Nat) (hidden :: Nat) (t :: TensorLike) a = LSTM
  { -- | Input features mapped to the packed output projections.
    lstmInputWeights :: t '[input, 4 * hidden] a,
    -- | Previous hidden features mapped to the packed output projections.
    lstmHiddenWeights :: t '[hidden, 4 * hidden] a,
    -- | Bias of the input projection.
    lstmInputBias :: t '[4 * hidden] a,
    -- | Bias of the recurrent projection.
    lstmHiddenBias :: t '[4 * hidden] a
  }
  deriving (Generic)

type instance WithTensor t (LSTM input hidden T.Tensor a) = LSTM input hidden t a

-- | Hidden and cell tensors for one layer and one batch.
-- Both tensors remain differentiable when represented by autograd tensors.
-- Initialize them explicitly, commonly with zeros; the library does not reset
-- or detach them between calls.
data LSTMState (batch :: Nat) (hidden :: Nat) (t :: TensorLike) a = LSTMState
  { -- | Hidden output passed to the next step.
    lstmHiddenState :: t '[batch, hidden] a,
    -- | Cell memory passed to the next step.
    lstmCellState :: t '[batch, hidden] a
  }
  deriving (Generic)

type instance WithTensor t (LSTMState batch hidden T.Tensor a) = LSTMState batch hidden t a

-- | Apply one LSTM step and update both state tensors.
-- Split the affine projection into gates @(i, f, g, o)@, then compute:
--
-- @
-- c' = sigmoid f * c + sigmoid i * tanh g
-- h' = sigmoid o * tanh c'
-- @
--
-- The returned output is @h'@. Both the initial hidden and cell tensors
-- receive gradients when included in the differentiation inputs.
lstmCell ::
  forall input hidden batch t a.
  ( KnownNat input,
    KnownNat hidden,
    KnownNat batch,
    KnownNat (4 * hidden),
    TensorOps t a,
    Num (TensorElement a),
    Num (t '[batch, 4 * hidden] a),
    Floating (t '[batch, hidden] a),
    E.Ord (t '[batch, hidden] a)
  ) =>
  LSTM input hidden t a ->
  t '[batch, input] a ->
  State (LSTMState batch hidden t a) (t '[batch, hidden] a)
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

-- | Apply one LSTM layer to @'[time, batch, input]@, returning hidden outputs
-- of shape @'[time, batch, hidden]@ and the final hidden/cell tensors through 'State'.
-- Parameters are shared across steps. An empty sequence preserves the initial
-- state. Carrying state between calls retains its differentiation history;
-- apply @AG.stopGrad@ explicitly at a truncated backpropagation boundary.
lstm ::
  forall input hidden batch time t a.
  ( KnownNat input,
    KnownNat hidden,
    KnownNat batch,
    KnownNat time,
    KnownNat (4 * hidden),
    TensorOps t a,
    Floating a,
    Floating (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  LSTM input hidden t a ->
  t '[time, batch, input] a ->
  State (LSTMState batch hidden t a) (t '[time, batch, hidden] a)
lstm = mapAccum lstmCell
