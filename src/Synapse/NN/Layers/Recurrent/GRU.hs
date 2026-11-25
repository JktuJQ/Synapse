{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Gated recurrent units with reset-after candidate projection.
module Synapse.NN.Layers.Recurrent.GRU (GRU (..), gruCell, gru) where

import Control.Monad.Trans.State.Strict (State, state)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, Nat, type (*))
import Synapse.NN.Activation (sigmoid)
import Synapse.NN.Layers.Recurrent.Sequence (sequenceHidden)
import Synapse.Tensor.Ops.Constraints (FloatingDType, OrdDType)
import Synapse.Tensor.Ops.Definition (TensorOps)
import Synapse.Tensor.Ops.Math (matmul)
import Synapse.Tensor.Ops.Shape (broadcast, split3)
import Synapse.Tensor.Representation (TensorLike)
import Synapse.Tensor.Representation qualified as T
import Synapse.Tensor.Structure (TensorStructure, WithTensor)

{- | Packed reset, update and candidate projections, in that order.

Input weights have shape @'[input, 3 * hidden]@; recurrent weights have shape
@'[hidden, 3 * hidden]@. Both biases have shape @'[3 * hidden]@.
The recurrent candidate bias is multiplied by the reset gate, so it cannot
be merged with the input candidate bias.
-}
data GRU (input :: Nat) (hidden :: Nat) (t :: TensorLike) a = GRU
    { gruInputWeights :: t '[input, 3 * hidden] a
    , gruHiddenWeights :: t '[hidden, 3 * hidden] a
    , gruInputBias :: t '[3 * hidden] a
    , gruHiddenBias :: t '[3 * hidden] a
    }
    deriving (Generic)

instance TensorStructure (GRU input hidden T.Tensor a) where
    type WithTensor t (GRU input hidden T.Tensor a) = GRU input hidden t a

{- | Apply one GRU step, reading and replacing the hidden tensor through 'State'.

For input projections @(xr, xz, xn)@ and recurrent projections @(hr, hz, hn)@:

@
r  = sigmoid (xr + hr)
z  = sigmoid (xz + hz)
n  = tanh (xn + r * hn)
h' = (1 - z) * n + z * h
@
-}
gruCell
    :: forall input hidden batch t a
     . ( KnownNat input
       , KnownNat hidden
       , KnownNat batch
       , KnownNat (3 * hidden)
       , TensorOps t a
       , FloatingDType a
       , OrdDType a
       , Num (t '[batch, 3 * hidden] a)
       , Floating (t '[batch, hidden] a)
       )
    => GRU input hidden t a
    -> t '[batch, input] a
    -> State (t '[batch, hidden] a) (t '[batch, hidden] a)
gruCell parameters input = state $ \previous ->
    let (xr, xz, xn) = split3 $ matmul input (gruInputWeights parameters) + broadcast (gruInputBias parameters)
        (hr, hz, hn) = split3 $ matmul previous (gruHiddenWeights parameters) + broadcast (gruHiddenBias parameters)
        reset = sigmoid $ xr + hr
        update = sigmoid $ xz + hz
        candidate = tanh $ xn + reset * hn
        next = (1 - update) * candidate + update * previous
     in (next, next)

-- | Apply one GRU layer to @'[time, batch, input]@.
gru
    :: forall input hidden batch time t a
     . ( KnownNat input
       , KnownNat hidden
       , KnownNat batch
       , KnownNat time
       , KnownNat (3 * hidden)
       , TensorOps t a
       , FloatingDType a
       , OrdDType a
       )
    => GRU input hidden t a
    -> t '[time, batch, input] a
    -> State (t '[batch, hidden] a) (t '[time, batch, hidden] a)
gru parameters =
    sequenceHidden
        (\((w, r), (b, c)) -> gruCell $ GRU w r b c)
        ( (gruInputWeights parameters, gruHiddenWeights parameters)
        , (gruInputBias parameters, gruHiddenBias parameters)
        )
