{- | This module implements several optimizers that are used in training.
-}


{- @FlexibleContexts@, @TypeFamilies@ are needed to
use @DType@ and define @Optimizer@ typeclass.
-}

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies     #-}


module Synapse.NN.Training.Optimizers
    ( -- @Optimizer@ typeclass
      
      Optimizer (OptimizerParameters, optimizerInitialParameters, optimizerUpdateStep)
    
      -- Optimizers

    , SGD (SGD, sgdMomentum, sgdNesterov)
    ) where


import Synapse.Tensors (DType, ElementwiseScalarOps((*.)))

import Synapse.Tensors.Mat (Mat)
import qualified Synapse.Tensors.Mat as M

import Synapse.NN.Layers.Initializers (ones)

import Data.Kind (Type)


-- | @Optimizer@ typeclass represents optimizer - algorithm that defines an update rule of neural network parameters.
class Num (DType optimizer) => Optimizer optimizer where
    -- | @OptimizerParameters@ represent optimizer-specific parameters that it needs to implement update rule.
    type OptimizerParameters optimizer :: Type

    -- | Returns initial state of optimizer-specific parameters for given variable.
    optimizerInitialParameters :: optimizer -> Mat (DType optimizer) -> OptimizerParameters optimizer
    
    -- | Performs the update step of optimizer.
    optimizerUpdateStep
        :: optimizer                                               -- ^ Optimizer itself.
        -> (Mat (DType optimizer), OptimizerParameters optimizer)  -- ^ Given parameter and current state of optimizer-specific parameters.
        -> (DType optimizer, Mat (DType optimizer))                -- ^ Learning rate and gradient of given parameter.
        -> (Mat (DType optimizer), OptimizerParameters optimizer)  -- ^ Updated parameter and a new state of optimizer-specific parameters.


-- | @SGD@ is a optimizer that implements stochastic gradient-descent algorithm.
data SGD a = SGD
    { sgdMomentum :: a     -- ^ Momentum coefficient.
    , sgdNesterov :: Bool  -- ^ Nesterov update rule.
    }

type instance DType (SGD a) = a

instance Num a => Optimizer (SGD a) where
    type OptimizerParameters (SGD a) = Mat a

    optimizerInitialParameters _ parameter = ones (M.size parameter)

    optimizerUpdateStep (SGD momentum nesterov) (parameter, velocity) (lr, gradient) = (parameter', velocity')
      where
        velocity' = velocity *. momentum - gradient *. lr

        parameter' = if nesterov
                     then parameter + velocity *. momentum - gradient *. lr
                     else parameter + velocity
