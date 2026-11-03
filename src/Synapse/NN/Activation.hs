{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Shape-preserving activation functions for tensors and element expressions.
-- Ordinary and differentiable values use the same functions.
-- Activations support ordinary function application and composition in an arbitrary arrow.
module Synapse.NN.Activation (relu, leakyRelu, sigmoid, tanh, sin, softplus) where

import Control.Arrow (Arrow, arr)
import qualified Synapse.Tensor.Ops.Elementwise as E
import Prelude hiding (sin, tanh)
import qualified Prelude as P

-- | Apply the rectified linear activation @relu x = max 0 x@ elementwise.
relu :: forall {arrow} a. (Arrow arrow, Num a, E.Ord a, E.Conditional a, E.Condition a ~ E.Comparison a) => arrow a a
relu = arr $ \x -> E.select (x E.<= 0) 0 x

-- | Apply @x@ when @x > 0@ and @slope * x@ otherwise, elementwise.
-- The first argument controls the slope for negative inputs.
-- For example, @leakyRelu 0.01 input@ retains one percent of each negative input.
leakyRelu :: forall {arrow} a. (Arrow arrow, Num a, E.Ord a, E.Conditional a, E.Condition a ~ E.Comparison a) => a -> arrow a a
leakyRelu slope = arr $ \x -> E.select (x E.> 0) x (slope * x)

-- | Apply the logistic sigmoid @sigmoid x = 1 / (1 + exp (-x))@ elementwise.
sigmoid :: forall {arrow} a. (Arrow arrow, Floating a, E.Ord a) => arrow a a
sigmoid = arr $ \x ->
  let numerator = exp $ E.min x 0
      complement = exp $ negate $ E.max x 0
   in numerator / (numerator + complement)

-- | Apply the hyperbolic tangent @tanh x = (exp x - exp (-x)) / (exp x + exp (-x))@ elementwise.
tanh :: forall {arrow} a. (Arrow arrow, Floating a) => arrow a a
tanh = arr P.tanh

-- | Apply the sine activation @sin x@ elementwise, with angles in radians.
sin :: forall {arrow} a. (Arrow arrow, Floating a) => arrow a a
sin = arr P.sin

-- | Apply @softplus x = log (1 + exp x)@ elementwise as a smooth approximation to 'relu'.
softplus :: forall {arrow} a. (Arrow arrow, Floating a, E.Ord a, E.Conditional a, E.Condition a ~ E.Comparison a) => arrow a a
softplus = arr $ \x ->
  let positive = x E.> 0
      exponential = exp $ E.select positive (-x) x
      correction = 2 * atanh (exponential / (2 + exponential))
   in E.select positive x 0 + correction
