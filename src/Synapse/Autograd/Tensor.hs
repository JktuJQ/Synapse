-- | Differentiable adapters for staged tensors and element expressions.
module Synapse.Autograd.Tensor
  ( Tensor (..),
    TensorElement (..),
  )
where

import Synapse.Autograd.Tensor.Ops.Index ()
import Synapse.Autograd.Tensor.Ops.Math ()
import Synapse.Autograd.Tensor.Ops.Reduction ()
import Synapse.Autograd.Tensor.Ops.Shape ()
import Synapse.Autograd.Tensor.Type (Tensor (..), TensorElement (..))
