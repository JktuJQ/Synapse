{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoDeepSubsumption #-}

-- | A single tanh recurrent layer and its stateful sequence traversal.
module Synapse.NN.Layers.Recurrent.RNN (RNN (..), rnnCell, rnn) where

import Control.Monad.Trans.State.Strict (State, state)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, Nat)
import Synapse.NN.Layers.Recurrent.Sequence (sequenceHidden)
import Synapse.Tensor.Ops.Constraints (FloatingDType, NumDType, OrdDType)
import Synapse.Tensor.Ops.Definition (TensorOps)
import Synapse.Tensor.Ops.Math (matmul)
import Synapse.Tensor.Ops.Shape (broadcast)
import Synapse.Tensor.Representation (TensorLike)
import qualified Synapse.Tensor.Representation as T
import Synapse.Tensor.Structure (TensorStructure, WithTensor)

{- | Input and recurrent projections for one recurrent layer.

Weights use @[input, hidden]@ and @[hidden, hidden]@ order.
Biases are separate and both have shape @'[hidden]@.
-}
data RNN (input :: Nat) (hidden :: Nat) (t :: TensorLike) a = RNN
    { rnnInputWeights :: t '[input, hidden] a
    , rnnHiddenWeights :: t '[hidden, hidden] a
    , rnnInputBias :: t '[hidden] a
    , rnnHiddenBias :: t '[hidden] a
    }
    deriving (Generic)

instance TensorStructure (RNN input hidden T.Tensor a) where
    type WithTensor t (RNN input hidden T.Tensor a) = RNN input hidden t a

-- | Apply @h' = tanh (x * W_input + b_input + h * W_hidden + b_hidden)@.
rnnCell
    :: forall input hidden batch t a
     . (KnownNat input, KnownNat hidden, KnownNat batch, TensorOps t a, NumDType a, Floating (t '[batch, hidden] a))
    => RNN input hidden t a
    -> t '[batch, input] a
    -> State (t '[batch, hidden] a) (t '[batch, hidden] a)
rnnCell parameters input = state $ \previous ->
    let next =
            tanh $
                matmul input (rnnInputWeights parameters)
                    + broadcast (rnnInputBias parameters)
                    + matmul previous (rnnHiddenWeights parameters)
                    + broadcast (rnnHiddenBias parameters)
     in (next, next)

{- | Apply one tanh recurrent layer to a time-major sequence.
Input shape is @'[time, batch, input]@ and output shape is @'[time, batch, hidden]@.
Supply the initial hidden tensor to 'Control.Monad.Trans.State.Strict.runState';
the action returns every hidden output and leaves the final hidden tensor in state.
An empty sequence preserves the initial state. Reuse it across chunks to carry
context, or supply a new state for an independent sequence.
-}
rnn
    :: forall input hidden batch time t a
     . (KnownNat input, KnownNat hidden, KnownNat batch, KnownNat time, TensorOps t a, FloatingDType a, OrdDType a)
    => RNN input hidden t a
    -> t '[time, batch, input] a
    -> State (t '[batch, hidden] a) (t '[time, batch, hidden] a)
rnn parameters =
    sequenceHidden
        (\((w, r), (b, c)) -> rnnCell $ RNN w r b c)
        ( (rnnInputWeights parameters, rnnHiddenWeights parameters)
        , (rnnInputBias parameters, rnnHiddenBias parameters)
        )
