{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Root mean square propagation.
module Synapse.NN.Optimizers.RMSprop (RMSprop (..)) where

import Synapse.Autograd.Type (Differentiable, ParametersConstraints)
import Synapse.NN.Optimizers.Type

-- | RMSprop without momentum or centering.
-- @v' = alpha * v + (1 - alpha) * gradient^2@ and @p' = p - lr * gradient / (sqrt v' + epsilon)@.
-- Typical values are @RMSprop 0.01 0.99 1e-8@.
data RMSprop = RMSprop
  { rmspropLearningRate :: Double,
    rmspropAlpha :: Double,
    rmspropEpsilon :: Double
  }
  deriving (Eq, Show)

instance (Differentiable p, ParametersConstraints Floating p) => Optimizer RMSprop p where
  type OptimizerState RMSprop p = p
  initialState _ p = updateParameters @Floating (\_ _ -> 0) p p
  optimizerStep (RMSprop lr alpha epsilon) p g previous =
    let square = updateParameters @Floating (\v gradient -> realToFrac alpha * v + realToFrac (1 - alpha) * gradient * gradient) previous g
        direction = updateParameters @Floating (\gradient v -> gradient / (sqrt v + realToFrac epsilon)) g square
     in (updateParameters @Floating (\parameter delta -> parameter - realToFrac lr * delta) p direction, square)
