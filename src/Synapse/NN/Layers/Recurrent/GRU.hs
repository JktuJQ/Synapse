{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | Gated recurrent units with reset-after candidate projection.
module Synapse.NN.Layers.Recurrent.GRU (GRU (..), gruCell, gru) where

import Control.Monad.Trans.State.Strict (State, state)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, Nat, type (*))
import Synapse.NN.Activation (sigmoid)
import Synapse.NN.Layers.Recurrent.Gates (split3)
import Synapse.Tensor.Ops (TensorOps)
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Math (matmul)
import Synapse.Tensor.Ops.Sequence (mapAccum)
import Synapse.Tensor.Ops.Shape (broadcast)
import Synapse.Tensor.Type (TensorElement, TensorLike, WithTensor)
import qualified Synapse.Tensor.Type as T

-- | Packed reset, update and candidate projections, in that order.
-- Input weights have shape @'[input, 3 * hidden]@; recurrent weights have shape
-- @'[hidden, 3 * hidden]@. Both biases have shape @'[3 * hidden]@.
-- The recurrent candidate bias is multiplied by the reset gate, so it cannot
-- be merged with the input candidate bias.
data GRU (input :: Nat) (hidden :: Nat) (t :: TensorLike) a = GRU
  { -- | Input features mapped to the packed output projections.
    gruInputWeights :: t '[input, 3 * hidden] a,
    -- | Previous hidden features mapped to the packed output projections.
    gruHiddenWeights :: t '[hidden, 3 * hidden] a,
    -- | Bias of the input projection.
    gruInputBias :: t '[3 * hidden] a,
    -- | Bias of the recurrent projection.
    gruHiddenBias :: t '[3 * hidden] a
  }
  deriving (Generic)

type instance WithTensor t (GRU input hidden T.Tensor a) = GRU input hidden t a

-- | Apply one GRU step, reading and replacing the hidden tensor through 'State'.
-- For input projections @(xr, xz, xn)@ and recurrent projections @(hr, hz, hn)@:
--
-- @
-- r  = sigmoid (xr + hr)
-- z  = sigmoid (xz + hz)
-- n  = tanh (xn + r * hn)
-- h' = (1 - z) * n + z * h
-- @
--
-- This reset-after form matches <https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html PyTorch's GRUCell>. The output is the new hidden state.
gruCell ::
  forall input hidden batch t a.
  ( KnownNat input,
    KnownNat hidden,
    KnownNat batch,
    KnownNat (3 * hidden),
    TensorOps t a,
    Num (TensorElement a),
    Num (t '[batch, 3 * hidden] a),
    Floating (t '[batch, hidden] a),
    E.Ord (t '[batch, hidden] a)
  ) =>
  GRU input hidden t a ->
  t '[batch, input] a ->
  State (t '[batch, hidden] a) (t '[batch, hidden] a)
gruCell parameters input = state $ \previous ->
  let (xr, xz, xn) = split3 $ matmul input (gruInputWeights parameters) + broadcast (gruInputBias parameters)
      (hr, hz, hn) = split3 $ matmul previous (gruHiddenWeights parameters) + broadcast (gruHiddenBias parameters)
      reset = sigmoid $ xr + hr
      update = sigmoid $ xz + hz
      candidate = tanh $ xn + reset * hn
      next = (1 - update) * candidate + update * previous
   in (next, next)

-- | Apply one GRU layer to @'[time, batch, input]@, returning every hidden output
-- with shape @'[time, batch, hidden]@ and preserving the final hidden tensor in 'State'.
-- The cell and sequence functions use the same parameters. An empty sequence
-- preserves the initial state; callers choose whether to carry it across chunks.
gru ::
  forall input hidden batch time t a.
  ( KnownNat input,
    KnownNat hidden,
    KnownNat batch,
    KnownNat time,
    KnownNat (3 * hidden),
    TensorOps t a,
    Floating a,
    Floating (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  GRU input hidden t a ->
  t '[time, batch, input] a ->
  State (t '[batch, hidden] a) (t '[time, batch, hidden] a)
gru = mapAccum gruCell
