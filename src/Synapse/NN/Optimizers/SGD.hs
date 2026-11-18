{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Stochastic gradient descent, momentum, and Nesterov updates.
module Synapse.NN.Optimizers.SGD (SGD (..), MomentumSGD (..)) where

import Control.Monad.Trans.State.Strict (get, put)

import Synapse.NN.Optimizers.Optimizer (Optimizer (..))
import Synapse.Tensor.Structure (FieldConstraints, TensorStructure)

import qualified Synapse.Tensor.Structure as Structure

-- | Stochastic gradient descent: @p' = p - lr * gradient@.
newtype SGD = SGD
    { sgdLearningRate :: Double
    }
    deriving (Eq, Show)

instance (TensorStructure p, FieldConstraints Fractional p) => Optimizer SGD p where
    type OptimizerState SGD p = ()
    initialState _ _ = ()
    optimizerStep (SGD lr) parameters gradients =
        pure $ Structure.zipWithFields @Fractional (\_ p g -> p - realToFrac lr * g) parameters gradients

{- | Momentum SGD: @v' = momentum * v + gradient@.

Nesterov uses @gradient + momentum * v'@ as the update direction.
Both variants update parameters as @p' = p - lr * direction@.
Supply a finite, nonnegative learning rate and a momentum coefficient in @[0, 1)@.
No dampening or weight decay is applied.
-}
data MomentumSGD = MomentumSGD
    { momentumSGDLearningRate :: Double
    , momentumSGDCoefficient :: Double
    , momentumSGDNesterov :: Bool
    }
    deriving (Eq, Show)

instance (TensorStructure p, FieldConstraints Fractional p) => Optimizer MomentumSGD p where
    type OptimizerState MomentumSGD p = p
    initialState _ p = Structure.zipWithFields @Fractional (\_ _ _ -> 0) p p
    optimizerStep (MomentumSGD lr coefficient nesterov) p g = do
        velocity <- get
        let nextVelocity =
                Structure.zipWithFields @Fractional (\_ v gradient -> realToFrac coefficient * v + gradient) velocity g
            direction =
                if nesterov
                    then
                        Structure.zipWithFields @Fractional
                            (\_ gradient v -> gradient + realToFrac coefficient * v)
                            g
                            nextVelocity
                    else nextVelocity
        put nextVelocity
        pure $
            Structure.zipWithFields @Fractional
                (\_ parameter gradient -> parameter - realToFrac lr * gradient)
                p
                direction
