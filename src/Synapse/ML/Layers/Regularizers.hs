{- | Provides collection of functions that impose penalties on parameters which is done by adding result to loss value.
-}


{- @TypeFamilies@ are needed to instantiate @DType@.
-}

{-# LANGUAGE TypeFamilies #-}


module Synapse.ML.Layers.Regularizers
    ( -- * @RegularizerFn@ type alias and @Regularizer@ newtype

      RegularizerFn
    , Regularizer (Regularizer, unRegularizer)

      -- * Regularizers

    , l1
    , l2
    ) where


import Synapse.LinearAlgebra (DType, ElementwiseScalarOps((*.)), SingletonOps(elementsSum))

import Synapse.Autograd (SymbolMat, Symbolic)


-- | @RegularizerFn@ type alias represents functions that impose penalties on parameters which is done by adding result of regularization to loss value.
type RegularizerFn a = SymbolMat a -> SymbolMat a

type instance DType (RegularizerFn a) = a


{- | @Regularizer@ newtype wraps @RegularizerFn@s - functions that impose penalties on parameters.

Every regularization function must return symbol of singleton matrix.
-}
newtype Regularizer a = Regularizer
    { unRegularizer :: RegularizerFn a  -- ^ Unwraps @Regularizer@ newtype.
    }

type instance DType (Regularizer a) = a


-- | Regularizers

-- | Applies a L1 regularization penalty (sum of absolute values of parameter multiplied by a coefficient).
l1 :: (Symbolic a, Num a) => a -> RegularizerFn a
l1 k mat = elementsSum (abs mat) *. k

-- | Applies a L1 regularization penalty (sum of squared values of parameter multiplied by a coefficient).
l2 :: (Symbolic a, Num a) => a -> RegularizerFn a
l2 k mat = elementsSum (mat * mat) *. k
