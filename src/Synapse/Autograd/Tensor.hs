-- | Differentiable adapters for staged tensors and element expressions.
-- Use the polymorphic operations from "Synapse.Tensor" with these wrappers inside a differentiation scope.
-- Supported operations include arithmetic, elementary floating-point functions, dot products, matrix multiplication, shape transformations, indexing, slicing, reductions, and normalization.
-- Comparisons produce ordinary boolean masks without gradient history; conditional selection propagates gradients to the selected value branches.
-- Element expressions retain gradients through arithmetic, selection, and conversion to scalar tensors.
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

import Synapse.Autograd.Tensor.Ops.Elementwise ()
import Synapse.Autograd.Tensor.Ops.Index ()
import Synapse.Autograd.Tensor.Ops.Math ()
import Synapse.Autograd.Tensor.Ops.Reduction ()
import Synapse.Autograd.Tensor.Ops.Shape ()
import Synapse.Autograd.Tensor.Type (Tensor (..), TensorElement (..))
