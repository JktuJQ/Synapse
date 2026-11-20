{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Adaptive moment estimation and decoupled weight decay.
module Synapse.NN.Optimizers.Adam (Adam (..), AdamW (..)) where

import Control.Monad.Trans.State.Strict (get, put)

import Synapse.NN.Optimizers.Optimizer (Optimizer (..))
import Synapse.Tensor.Ops.Constraints (FloatingDType)
import Synapse.Tensor.Structure (FieldConstraints, TensorStructure)

import qualified Synapse.Tensor.Structure as Structure

import qualified Synapse.Tensor as T

-- | Correct moments using a tensor-valued step count converted to the moment element type.
class Floating a => AdamParameter a where
    -- | Divide an accumulated moment by @1 - beta^step@.
    correctMoment
        :: Double
        -> T.Tensor '[] Int
        -> a
        -> a

instance
    (T.Shape sh, FloatingDType a, T.FromIntegral (T.Tensor '[] Int) (T.Tensor '[] a))
    => AdamParameter (T.Tensor sh a)
    where
    correctMoment beta count moment =
        moment / (1 - realToFrac beta ** T.broadcast (T.fromIntegral count :: T.Tensor '[] a))

{- | Adam optimizer with learning rate, first and second moment decay, and numerical epsilon.

The update is @p' = p - lr * mHat / (sqrt vHat + epsilon)@.
Typical values are @Adam 0.001 0.9 0.999 1e-8@.
Moments start at zero and are corrected by @mHat = m / (1 - beta1^step)@ and @vHat = v / (1 - beta2^step)@.
State is @(step, (firstMoment, secondMoment))@; the step is a scalar integer tensor, initially zero.
Supply a nonnegative learning rate, decay coefficients in @[0, 1)@, and a positive epsilon.
Parameter elements must support floating-point operations and conversion from the integer step count.
-}
data Adam = Adam
    { adamLearningRate :: Double
    , adamBeta1 :: Double
    , adamBeta2 :: Double
    , adamEpsilon :: Double
    }
    deriving (Eq, Show)

instance (TensorStructure p, FieldConstraints AdamParameter p) => Optimizer Adam p where
    type OptimizerState Adam p = (T.Tensor '[] Int, (p, p))
    initialState _ p =
        let zero = Structure.zipWithFields @AdamParameter (\_ _ _ -> 0) p p
         in (0, (zero, zero))
    optimizerStep (Adam lr beta1 beta2 epsilon) p g = do
        (count, (first, second)) <- get
        let nextCount = count + 1
            nextFirst =
                Structure.zipWithFields @AdamParameter
                    (\_ m gradient -> realToFrac beta1 * m + realToFrac (1 - beta1) * gradient)
                    first
                    g
            nextSecond =
                Structure.zipWithFields @AdamParameter
                    (\_ v gradient -> realToFrac beta2 * v + realToFrac (1 - beta2) * gradient * gradient)
                    second
                    g
            direction =
                Structure.zipWithFields @AdamParameter
                    ( \_ m v ->
                        correctMoment beta1 nextCount m
                            / (sqrt (correctMoment beta2 nextCount v) + realToFrac epsilon)
                    )
                    nextFirst
                    nextSecond
            updated =
                Structure.zipWithFields @AdamParameter
                    (\_ parameter delta -> parameter - realToFrac lr * delta)
                    p
                    direction
        put (nextCount, (nextFirst, nextSecond))
        pure updated

{- | Adam with decoupled weight decay applied to every parameter tensor, including biases.

After the Adam update, subtract @lr * weightDecay * p@ using the original parameters.
State and initialization are the same as for t'Adam'; supply a finite, nonnegative weight decay.
Typical values are @AdamW (Adam 0.001 0.9 0.999 1e-8) 0.01@.
-}
data AdamW = AdamW
    { adamWAdam :: Adam
    , adamWWeightDecay :: Double
    }
    deriving (Eq, Show)

instance (TensorStructure p, FieldConstraints AdamParameter p) => Optimizer AdamW p where
    type OptimizerState AdamW p = OptimizerState Adam p
    initialState (AdamW adam _) = initialState adam
    optimizerStep (AdamW adam decay) p g = do
        updated <- optimizerStep adam p g
        pure $
            Structure.zipWithFields @AdamParameter
                (\_ new old -> new - realToFrac (adamLearningRate adam * decay) * old)
                updated
                p
