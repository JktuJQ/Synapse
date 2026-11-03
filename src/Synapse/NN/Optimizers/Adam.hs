{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Adaptive moment estimation and decoupled weight decay.
module Synapse.NN.Optimizers.Adam (Adam (..), AdamW (..)) where

import Control.Monad.Trans.State.Strict (get, put)
import Synapse.Autograd.Type (Differentiable, ParametersConstraints)
import Synapse.NN.Optimizers.Type
import qualified Synapse.Tensor as T

-- | Adam hyperparameters: learning rate, first and second moment decay, and numerical epsilon.
-- The update is @p' = p - lr * mHat / (sqrt vHat + epsilon)@.
-- Typical values are @Adam 0.001 0.9 0.999 1e-8@.
-- State consists of a scalar tensor step count and two moment records, and supports prepared backend execution.
-- Parameter tensors may contain Float or Double elements.
data Adam = Adam
  { adamLearningRate :: Double,
    adamBeta1 :: Double,
    adamBeta2 :: Double,
    adamEpsilon :: Double
  }
  deriving (Eq, Show)

-- | Adam with decoupled weight decay applied to every parameter tensor.
-- Typical values are @AdamW (Adam 0.001 0.9 0.999 1e-8) 0.01@.
data AdamW = AdamW
  { adamWAdam :: Adam,
    adamWWeightDecay :: Double
  }
  deriving (Eq, Show)

class (Floating a) => AdamParameter a where
  correctMoment :: Double -> T.Tensor '[] Int -> a -> a

instance (T.Shape sh) => AdamParameter (T.Tensor sh Float) where
  correctMoment beta count moment = moment / (1 - realToFrac beta ** T.broadcast (T.fromIntegral count :: T.Tensor '[] Float))

instance (T.Shape sh) => AdamParameter (T.Tensor sh Double) where
  correctMoment beta count moment = moment / (1 - realToFrac beta ** T.broadcast (T.fromIntegral count :: T.Tensor '[] Double))

instance (Differentiable p, ParametersConstraints AdamParameter p) => Optimizer Adam p where
  type OptimizerState Adam p = (T.Tensor '[] Int, (p, p))
  initialState _ p =
    let zero = updateParameters @AdamParameter (\_ _ -> 0) p p
     in (0, (zero, zero))
  optimizerStep (Adam lr beta1 beta2 epsilon) p g = do
    (count, (first, second)) <- get
    let nextCount = count + 1
        nextFirst = updateParameters @AdamParameter (\m gradient -> realToFrac beta1 * m + realToFrac (1 - beta1) * gradient) first g
        nextSecond = updateParameters @AdamParameter (\v gradient -> realToFrac beta2 * v + realToFrac (1 - beta2) * gradient * gradient) second g
        direction = updateParameters @AdamParameter (\m v -> correctMoment beta1 nextCount m / (sqrt (correctMoment beta2 nextCount v) + realToFrac epsilon)) nextFirst nextSecond
        updated = updateParameters @AdamParameter (\parameter delta -> parameter - realToFrac lr * delta) p direction
    put (nextCount, (nextFirst, nextSecond))
    pure updated

instance (Differentiable p, ParametersConstraints AdamParameter p) => Optimizer AdamW p where
  type OptimizerState AdamW p = OptimizerState Adam p
  initialState (AdamW adam _) = initialState adam
  optimizerStep (AdamW adam decay) p g = do
    updated <- optimizerStep adam p g
    pure $ updateParameters @AdamParameter (\new old -> new - realToFrac (adamLearningRate adam * decay) * old) updated p
