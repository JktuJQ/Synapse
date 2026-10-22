-- | Differentiable adapters for staged tensors and element expressions.
-- Use the polymorphic operations from "Synapse.Tensor" with these wrappers inside a differentiation scope.
-- Supported operations include arithmetic, elementary floating-point functions, dot products, matrix multiplication, shape transformations, indexing, slicing, reductions, and normalization.
-- Arbitrary element callbacks such as @map@ and @zipWith@, and element-level control flow, currently have no differentiable instances.
--
-- A t'Tensor' retains its static shape; full reductions return a scalar tensor.
-- A t'TensorElement' represents a single staged element, including the result of tensor indexing.
-- Neither representation executes the tensor graph during differentiation.
-- Extrema reductions divide the gradient equally between tied extrema.
-- Index-valued @argmin@ and @argmax@ results propagate zero gradients to their inputs.
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
