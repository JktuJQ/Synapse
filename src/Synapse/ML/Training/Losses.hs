{- | Provides collection of functions that are used to as a reference of what needs to be minimised during training.

@LossFn@ type alias represents those functions, and @Synapse@ offers a variety of them.
-}


{- @TypeFamilies@ are needed to instantiate @DType@.
-}

{-# LANGUAGE TypeFamilies #-}


module Synapse.ML.Training.Losses
    ( -- * @LossFn@ type alias and @Loss@ newtype

      LossFn
    
    , Loss (Loss, unLoss)

      -- * Regression losses
    
    , mse
    , msle
    , mae
    , mape
    , logcosh
    ) where


import Synapse.LinearAlgebra (DType, ElementwiseScalarOps((+.), (*.), (**.)), SingletonOps(mean))

import Synapse.Autograd (SymbolMat, Symbolic)


-- | @LossFn@ type alias represents functions that are able to provide a reference of what relation between matrices needs to be minimised.
type LossFn a = SymbolMat a -> SymbolMat a -> SymbolMat a

type instance DType (LossFn a) = a


{- | @Loss@ newtype wraps @LossFn@s - differentiable functions that are able to provide a reference of what relation between matrices needs to be minimised.

Every loss function must return symbol of singleton matrix.
-}
newtype Loss a = Loss 
    { unLoss :: LossFn a  -- ^ Unwraps @Loss@ newtype.
    }

type instance DType (Loss a) = a


-- Regression losses

-- | Computes the mean of squares of errors.
mse :: (Symbolic a, Floating a) => LossFn a
mse true predicted = mean $ (true - predicted) **. 2.0

-- | Computes the mean squared logarithmic error.
msle :: (Symbolic a, Floating a) => LossFn a
msle true predicted = mean $ (log (true +. 1) - log (predicted +. 1)) **. 2.0

-- | Computes the mean of absolute error.
mae :: (Symbolic a, Floating a) => LossFn a
mae true predicted = mean $ abs (true - predicted)

-- | Computes the mean absolute percentage error.
mape :: (Symbolic a, Floating a) => LossFn a
mape true predicted = mean (abs (true - predicted) / true) *. 100

-- | Computes the logarithm of the hyperbolic cosine of the error.
logcosh :: (Symbolic a, Floating a) => LossFn a
logcosh true predicted = mean $ log $ cosh (true - predicted)
