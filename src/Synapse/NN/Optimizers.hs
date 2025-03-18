{- | This module implements several optimizers that are used in training.
-}


-- 'TypeFamilies' are needed to use 'DType' and define 'Optimizer' typeclass.
{-# LANGUAGE TypeFamilies #-}


module Synapse.NN.Optimizers
    ( -- * 'Optimizer' typeclass
      
      Optimizer (OptimizerParameters, optimizerInitialParameters, optimizerUpdateStep)

    , optimizerUpdateParameters
    
      -- * Optimizers

    , SGD (SGD, sgdMomentum, sgdNesterov)
    ) where


import Synapse.Tensors (DType, ElementwiseScalarOps((*.)))

import Synapse.Tensors.Mat (Mat)
import qualified Synapse.Tensors.Mat as M

import Synapse.Autograd (Symbolic, Symbol(unSymbol), SymbolMat, Gradients, wrt)

import Synapse.NN.Layers.Initializers (zeroes)

import Data.Kind (Type)


-- | 'Optimizer' typeclass represents optimizer - algorithm that defines an update rule of neural network parameters.
class Optimizer optimizer where
    -- | 'OptimizerParameters' represent optimizer-specific parameters that it needs to implement update rule.
    type OptimizerParameters optimizer a :: Type

    -- | Returns initial state of optimizer-specific parameters for given variable.
    optimizerInitialParameters :: Num a => optimizer a -> Mat a -> OptimizerParameters optimizer a
    
    -- | Performs the update step of optimizer.
    optimizerUpdateStep
        :: Num a
        => optimizer a                               -- ^ Optimizer itself.
        -> (a, Mat a)                                -- ^ Learning rate and gradient of given parameter.
        -> (Mat a, OptimizerParameters optimizer a)  -- ^ Given parameter and current state of optimizer-specific parameters.
        -> (Mat a, OptimizerParameters optimizer a)  -- ^ Updated parameter and a new state of optimizer-specific parameters.

-- | 'optimizerUpdateParameters' function updates whole model using optimizer by performing 'optimizerUpdateStep' for every parameter.
optimizerUpdateParameters
    :: (Symbolic a, Optimizer optimizer)
    => optimizer a                                       -- ^ Optimizer itself.
    -> (a, Gradients (Mat a))                            -- ^ Learning rate and gradients of all parameters. 
    -> [(SymbolMat a, OptimizerParameters optimizer a)]  -- ^ Given parameters and current state of optimizer-specific parameters.
    -> [(Mat a, OptimizerParameters optimizer a)]        -- ^ Updated parameters and a new state of optimizer-specific parameters.
optimizerUpdateParameters _ _ [] = []
optimizerUpdateParameters optimizer (lrValue, gradients) ((parameter, optimizerParameter):xs) =
    optimizerUpdateStep optimizer (lrValue, unSymbol $ gradients `wrt` parameter) (unSymbol parameter, optimizerParameter)
    : optimizerUpdateParameters optimizer (lrValue, gradients) xs


-- | 'SGD' is a optimizer that implements stochastic gradient-descent algorithm.
data SGD a = SGD
    { sgdMomentum :: a     -- ^ Momentum coefficient.
    , sgdNesterov :: Bool  -- ^ Nesterov update rule.
    } deriving (Eq, Show)

type instance DType (SGD a) = a

instance Optimizer SGD where
    type OptimizerParameters SGD a = Mat a

    optimizerInitialParameters _ parameter = zeroes (M.size parameter)

    optimizerUpdateStep (SGD momentum nesterov) (lr, gradient) (parameter, velocity) = (parameter', velocity')
      where
        velocity' = velocity *. momentum - gradient *. lr

        parameter' = if nesterov
                     then parameter + velocity' *. momentum - gradient *. lr
                     else parameter + velocity'
