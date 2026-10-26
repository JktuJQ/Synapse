-- | Differentiable staged tensors with statically known shapes.
-- Use the polymorphic operations from "Synapse.Tensor" inside a differentiation scope.
-- The tensor constructor preserves history when accessing the variable provided by "Synapse.Autograd".
-- Differentiation entry points accept tensors and tensor records rather than standalone element expressions.
module Synapse.Autograd.Tensor
  ( Tensor (..),
  )
where

import Synapse.Autograd.Tensor.Ops ()
import Synapse.Autograd.Tensor.Ops.Elementwise ()
import Synapse.Autograd.Tensor.Type (Tensor (..))
