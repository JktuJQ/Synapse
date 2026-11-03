{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

-- | A single tanh recurrent layer and its stateful sequence traversal.
module Synapse.NN.Layers.Recurrent.RNN (RNN (..), rnnCell, rnn) where

import Control.Monad.Trans.State.Strict (State, state)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, Nat)
import Synapse.Tensor.Ops (TensorOps)
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Math (matmul)
import Synapse.Tensor.Ops.Sequence (mapAccum)
import Synapse.Tensor.Ops.Shape (broadcast)
import Synapse.Tensor.Type (TensorElement, TensorLike, WithTensor)
import qualified Synapse.Tensor.Type as T

-- | Input and recurrent projections for one recurrent layer.
-- Weights use @[input, hidden]@ and @[hidden, hidden]@ order.
-- Biases are separate and both have shape @'[hidden]@.
data RNN (input :: Nat) (hidden :: Nat) (t :: TensorLike) a = RNN
  { -- | Input features mapped to the packed output projections.
    rnnInputWeights :: t '[input, hidden] a,
    -- | Previous hidden features mapped to the packed output projections.
    rnnHiddenWeights :: t '[hidden, hidden] a,
    -- | Bias of the input projection.
    rnnInputBias :: t '[hidden] a,
    -- | Bias of the recurrent projection.
    rnnHiddenBias :: t '[hidden] a
  }
  deriving (Generic)

type instance WithTensor t (RNN input hidden T.Tensor a) = RNN input hidden t a

-- | Apply @h' = tanh (x * W_input + b_input + h * W_hidden + b_hidden)@.
-- Read and replace the hidden state through 'State'. The returned output is @h'@.
-- State tensors participate in differentiation just like the input and weights.
rnnCell ::
  forall input hidden batch t a.
  ( KnownNat input,
    KnownNat hidden,
    KnownNat batch,
    TensorOps t a,
    Num (TensorElement a),
    Floating (t '[batch, hidden] a)
  ) =>
  RNN input hidden t a ->
  t '[batch, input] a ->
  State (t '[batch, hidden] a) (t '[batch, hidden] a)
rnnCell parameters input = state $ \previous ->
  let next =
        tanh $
          matmul input (rnnInputWeights parameters)
            + broadcast (rnnInputBias parameters)
            + matmul previous (rnnHiddenWeights parameters)
            + broadcast (rnnHiddenBias parameters)
   in (next, next)

-- | Apply one tanh recurrent layer to a time-major sequence.
-- Input shape is @'[time, batch, input]@ and output shape is @'[time, batch, hidden]@.
-- Supply the initial hidden tensor to 'Control.Monad.Trans.State.Strict.runState';
-- the action returns every hidden output and leaves the final hidden tensor in state.
-- An empty sequence preserves the initial state. Reuse it across chunks to carry
-- context, or supply a new state for an independent sequence.
rnn ::
  forall input hidden batch time t a.
  ( KnownNat input,
    KnownNat hidden,
    KnownNat batch,
    KnownNat time,
    TensorOps t a,
    Floating a,
    Floating (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  RNN input hidden t a ->
  t '[time, batch, input] a ->
  State (t '[batch, hidden] a) (t '[time, batch, hidden] a)
rnn = mapAccum rnnCell
