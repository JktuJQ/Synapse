{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Stochastic gradient descent with optional momentum and Nesterov updates.
module Synapse.NN.Optimizers.SGD (SGD (..), Momentum (..)) where

import Synapse.Autograd.Type (Differentiable, ParametersConstraints)
import Synapse.NN.Optimizers.Type

-- | Stochastic gradient descent: @p' = p - learningRate * gradient@.
-- The caller supplies a finite, nonnegative learning rate.
-- Hyperparameters are used without runtime validation.
newtype SGD = SGD
  { -- | Step size applied to each gradient.
    sgdLearningRate :: Double
  }
  deriving (Eq, Show)

instance (Differentiable p, ParametersConstraints Fractional p) => Optimizer SGD p where
  type OptimizerState SGD p = ()
  initialState _ _ = ()
  optimizerStep (SGD lr) parameters gradients () =
    (updateParameters @Fractional (\p g -> p - realToFrac lr * g) parameters gradients, ())

-- | Momentum SGD: @v' = momentum * v + gradient@.
-- Nesterov uses @gradient + momentum * v'@ as the update direction.
-- Initial velocity is zero; no dampening or weight decay is applied.
data Momentum = Momentum
  { momentumLearningRate :: Double,
    momentumCoefficient :: Double,
    momentumNesterov :: Bool
  }
  deriving (Eq, Show)

instance (Differentiable p, ParametersConstraints Fractional p) => Optimizer Momentum p where
  type OptimizerState Momentum p = p
  initialState _ p = updateParameters @Fractional (\_ _ -> 0) p p
  optimizerStep (Momentum lr coefficient nesterov) p g velocity =
    let nextVelocity = updateParameters @Fractional (\v gradient -> realToFrac coefficient * v + gradient) velocity g
        direction = if nesterov then updateParameters @Fractional (\gradient v -> gradient + realToFrac coefficient * v) g nextVelocity else nextVelocity
     in (updateParameters @Fractional (\parameter gradient -> parameter - realToFrac lr * gradient) p direction, nextVelocity)
