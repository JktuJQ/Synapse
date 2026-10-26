{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Evaluation and reverse-mode differentiation of tensor functions.
-- Tensor records preserve their structure and register each tensor as an independent input.
-- Values and gradients remain staged expressions; execute them through "Synapse.Tensor".
module Synapse.Autograd.Execution (evaluate, backprop, gradient, pullback, stopGrad, constant) where

import qualified Numeric.Backprop.Explicit as BPE
import qualified Synapse.Autograd.Execution.Reverse as Reverse
import Synapse.Autograd.Execution.Variable (GradientView (..), Values (..), Variables, constants, ones, pack, stopped)
import Synapse.Autograd.Tensor ()
import Synapse.Autograd.Type (Differentiable, WithGrad)
import Synapse.Autograd.Variable (Context)
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import qualified Synapse.Tensor.Type as T

-- | Evaluate a tensor function with constant inputs, without a reverse pass.
-- This builds staged tensor expressions without executing the backend.
evaluate ::
  forall a b.
  (Differentiable a, Differentiable b) =>
  (forall s. (Context s) => WithGrad s a -> WithGrad s b) ->
  a ->
  b
evaluate f input = rebuild @T.Tensor $ BPE.evalBP0 result
  where
    result :: forall s. (Context s) => BPE.BVar s (Leaves T.Tensor b)
    result = pack (values @(Leaves T.Tensor b)) (gradientFields @s @b (f @s (constant @s @a input)))

-- | Return the output and the gradient of the sum of its tensor elements.
-- Unused inputs receive zero tensors with their original shapes.
backprop ::
  forall a b.
  (Differentiable a, Differentiable b) =>
  (forall s. (Context s) => WithGrad s a -> WithGrad s b) ->
  a ->
  (b, a)
backprop f input =
  let (output, backward) = pullback @a @b f input
      seed = rebuild @T.Tensor (ones (values @(Leaves T.Tensor b)) (flatten @T.Tensor output))
   in (output, backward seed)

-- | Return only the input gradient computed by 'backprop'.
gradient ::
  forall a b.
  (Differentiable a, Differentiable b) =>
  (forall s. (Context s) => WithGrad s a -> WithGrad s b) ->
  a ->
  a
gradient f = snd . backprop @a @b f

-- | Return the output and a reusable vector-Jacobian product.
-- Output seeds must preserve the output structure and tensor shapes.
pullback ::
  forall a b.
  (Differentiable a, Differentiable b) =>
  (forall s. (Context s) => WithGrad s a -> WithGrad s b) ->
  a ->
  (b, b -> a)
pullback f input =
  let wrapped :: forall s. (Context s) => Variables s (Leaves T.Tensor a) -> Variables s (Leaves T.Tensor b)
      wrapped = gradientFields @s @b . f @s . gradientRecord @s @a
      (output, backward) =
        Reverse.pullback
          (values @(Leaves T.Tensor a))
          (values @(Leaves T.Tensor b))
          wrapped
          (flatten @T.Tensor input)
   in (rebuild @T.Tensor output, rebuild @T.Tensor . backward . flatten @T.Tensor)

-- | Introduce staged tensors without registering them as differentiation inputs.
-- Constants can change between backend calls when supplied as computation inputs.
constant :: forall s a. (Differentiable a) => a -> WithGrad s a
constant = gradientRecord @s @a . constants @s (values @(Leaves T.Tensor a)) . flatten @T.Tensor

-- | Preserve tensor values while discarding their incoming differentiation history.
-- Other uses of the original tensors retain their histories.
stopGrad :: forall s a. (Differentiable a) => WithGrad s a -> WithGrad s a
stopGrad = gradientRecord @s @a . stopped (values @(Leaves T.Tensor a)) . gradientFields @s @a
