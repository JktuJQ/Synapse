{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Evaluation and reverse-mode differentiation of functions.
-- Tensor values and gradients remain staged expressions; execute them separately through "Synapse.Tensor".
module Synapse.Autograd.Execution
  ( evaluate,
    backprop,
    gradient,
    pullback,
    stopGrad,
    constant,
  )
where

import qualified Numeric.Backprop.Explicit as BPE
import Synapse.Autograd.Differentiable (Differentiable (..))
import qualified Synapse.Autograd.Execution.Reverse as Reverse
import Synapse.Autograd.Execution.Traversal (constants, ones, pack, stopped)
import Synapse.Autograd.Tensor ()
import Synapse.Autograd.Type (Context)

-- | Evaluate a function without running a reverse pass.
-- Inputs are introduced as constants, allowing operations to skip recording gradient history.
-- The result uses the ordinary representation corresponding to the function's output.
-- Evaluating a tensor function here builds its expression without executing the tensor graph.
evaluate ::
  forall a b.
  (Differentiable a, Differentiable b) =>
  (forall s. (Context s) => WithGrad s a -> WithGrad s b) ->
  a ->
  b
evaluate f value = BPE.evalBP0 (pack (structure @b) (f (constant value)))

-- | Return the function's value and its gradient with respect to the input, in that order.
-- Each output leaf is seeded using its @Backpropagatable.one@ implementation with its value as a reference.
-- For the standard numeric and tensor instances, this differentiates the sum of all output elements.
-- Input leaves that do not affect the result receive their reference-shaped zero gradient.
-- Records retain their ordinary structure in both the value and the gradient.
backprop ::
  forall a b.
  (Differentiable a, Differentiable b) =>
  (forall s. (Context s) => WithGrad s a -> WithGrad s b) ->
  a ->
  (b, a)
backprop f input =
  let (value, backward) = pullback f input
   in (value, backward (ones (structure @b) value))

-- | Return only the input gradient computed by 'backprop'.
-- The output seed follows the same convention as 'backprop'.
gradient ::
  forall a b.
  (Differentiable a, Differentiable b) =>
  (forall s. (Context s) => WithGrad s a -> WithGrad s b) ->
  a ->
  a
gradient f = snd . backprop f

-- | Return the function's value and a function that propagates an output gradient back to the input.
-- For a function @f@ evaluated at @x@, the returned function computes @J_f(x)^T seed@.
-- The seed must have the same structure and tensor shapes as the output value.
-- The input gradient has the same structure and tensor shapes as the original input.
-- The returned function can be applied repeatedly with different seeds to the same forward computation.
pullback ::
  forall a b.
  (Differentiable a, Differentiable b) =>
  (forall s. (Context s) => WithGrad s a -> WithGrad s b) ->
  a ->
  (b, b -> a)
pullback = Reverse.pullback (structure @a) (structure @b)

-- | Introduce an ordinary value as a constant in a differentiation scope.
-- Constants are not registered as inputs and receive no gradient contributions.
-- Structured values preserve their field representations, and tensor expressions remain staged.
-- Operations involving constants can still propagate gradients to their other, variable inputs.
-- Use 'stopGrad' instead to cut the history of a value already inside the differentiation scope.
constant :: forall s a. (Differentiable a) => a -> WithGrad s a
constant = constants @s (structure @a)

-- | Preserve a value while preventing differentiation through its history.
-- This cuts only paths through the returned value; other uses of the original value retain their history.
-- Tensor expressions remain staged.
--
-- @
-- -- Returns (9, 3): only the first factor contributes to the derivative.
-- backprop (\x -> x * stopGrad x) (3 :: Double)
-- @
stopGrad :: forall s a. (Differentiable a) => WithGrad s a -> WithGrad s a
stopGrad = stopped (structure @a)
