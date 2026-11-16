{- | Reverse-mode differentiation of tensor functions.

Functions use the operations from "Synapse.Tensor" with differentiable 'Tensor' values.
Differentiation returns ordinary staged tensors; execute their values and gradients with @T.run@.

This module is intended to be imported qualified:

@
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
@

= Values and gradients

'backprop' returns the function result and the gradient with respect to its input.
For example, @f(x) = x * x@ has derivative @2 * x@:

@
let input = T.full \@'[] (3 :: Double)
    (value, derivative) = T.run T.CPU $ AG.backprop (\\x -> x * x) input

T.toList value       -- [9.0]
T.toList derivative  -- [6.0]
@

The function passed to 'backprop' must work in any differentiation scope @s@.
'Context' supplies that scope; the type parameter prevents variables from different scopes being mixed.
Tensor functions polymorphic through @T.TensorOps@ can be used with both ordinary and differentiable tensors.

* 'evaluate' returns the result with constant inputs, without computing gradients.
* 'backprop' returns the result and the input gradient.
* 'gradient' returns only the input gradient.
* 'pullback' returns the result and a function mapping output gradients to input gradients.

For non-scalar results, 'backprop' and 'gradient' differentiate the sum of all output tensor elements.
Use 'pullback' to supply other output gradients:

@
let input = T.fromList \@'[3] [1, 2, 3 :: Double]
    (_, backward) = AG.pullback (\\x -> x * x) input
    seed = T.fromList \@'[3] [0, 1, 0 :: Double]

T.toList $ T.run T.CPU $ backward seed  -- [0.0, 4.0, 0.0]
@

The seed must have the same tensor shapes and structure as the result.
The pullback computes @J_f(input)^T * seed@ and can be applied to different seeds at the same input.

= Tensor structures

Inputs and outputs may be individual tensors, tuples, or parameter records.
'WithGrad' replaces each tensor in a structure with its differentiable representation,
while preserving its shape, element type, and surrounding structure.
'Differentiable' is inferred from the tensor structure and its gradient accumulation instances.

For a parameter record @Parameters t@, derive @Generic@ and define its @T.TensorStructure@ instance,
including the correspondence @WithTensor t (Parameters T.Tensor) = Parameters t@.
The instance can use the generic defaults described in "Synapse.Tensor.Structure".
No separate 'Differentiable' instance is needed.
A gradient has the same structure as the original input; unused tensor fields receive zero gradients.
Each tensor field is registered independently, and repeated uses of a variable accumulate their contributions.

= Constants and gradient boundaries

'constant' introduces ordinary tensors into a differentiated function without making them gradient inputs.
For example, a loss can close over training data and differentiate only its parameter argument.
'stopGrad' preserves a computed value but stops gradients through that particular use:

@
let input = T.full \@'[] (3 :: Double)
    derivative = AG.gradient (\\x -> x * AG.stopGrad x) input

T.toList $ T.run T.CPU derivative  -- [3.0]
@

Other uses of the same variable retain their differentiation history.
Both functions also accept tensor structures.

= Custom operations

Compositions of existing tensor operations are differentiated automatically.
To supply a custom reverse rule, use @withVJP@, @withCachedVJP@, or @customVJP@ from "Synapse.Tensor.Ops".
-}
module Synapse.Autograd
    ( -- * Differentiable tensors and structures
      Context
    , Tensor
    , Differentiable
    , WithGrad

      -- * Evaluation and differentiation
    , evaluate
    , backprop
    , gradient
    , pullback

      -- * Gradient boundaries
    , stopGrad
    , constant
    )
where

import Synapse.Autograd.Execution
    ( Differentiable
    , backprop
    , constant
    , evaluate
    , gradient
    , pullback
    -- \* Gradient boundaries
    , stopGrad
    )
import Synapse.Autograd.Representation (Context)
import Synapse.Autograd.Tensor (Tensor, WithGrad)
