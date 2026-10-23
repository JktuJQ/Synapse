{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Shape-preserving activation functions for tensors and element expressions.
-- Ordinary and differentiable values use the same functions.
module Synapse.NN.Activation (relu, leakyRelu, sigmoid, tanh, softplus) where

import qualified Synapse.Tensor.Ops.Elementwise as E
import Prelude hiding (tanh)
import qualified Prelude as P

-- | Apply the rectified linear activation @relu x = max 0 x@ elementwise.
relu :: (Num a, E.Ord a, E.Conditional a, E.Condition a ~ E.Comparison a) => a -> a
relu x = E.select (x E.<= 0) 0 x

-- | Apply @x@ when @x > 0@ and @slope * x@ otherwise, elementwise.
-- The first argument controls the slope for negative inputs.
-- For example, @leakyRelu 0.01 input@ retains one percent of each negative input.
leakyRelu :: (Num a, E.Ord a, E.Conditional a, E.Condition a ~ E.Comparison a) => a -> a -> a
leakyRelu slope x = E.select (x E.> 0) x (slope * x)

-- | Apply the logistic sigmoid @sigmoid x = 1 / (1 + exp (-x))@ elementwise.
sigmoid :: (Floating a, E.Ord a, E.Conditional a, E.Condition a ~ E.Comparison a) => a -> a
sigmoid x =
  let positive = x E.> 0
      exponential = exp $ E.select positive (-x) x
      denominator = 1 + exponential
   in E.select positive (recip denominator) (exponential / denominator)

-- | Apply the hyperbolic tangent @tanh x = (exp x - exp (-x)) / (exp x + exp (-x))@ elementwise.
tanh :: (Floating a) => a -> a
tanh = P.tanh

-- | Apply @softplus x = log (1 + exp x)@ elementwise as a smooth approximation to 'relu'.
softplus :: (Floating a, E.Ord a, E.Conditional a, E.Condition a ~ E.Comparison a) => a -> a
softplus x =
  let positive = x E.> 0
      exponential = exp $ E.select positive (-x) x
      correction = 2 * atanh (exponential / (2 + exponential))
   in E.select positive x 0 + correction
