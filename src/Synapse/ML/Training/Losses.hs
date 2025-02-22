{- | Provides collection of functions that are used to as a reference of what needs to be minimised during training.

@LossFn@ type alias represents those functions, and @Synapse@ provides a variety of them.
-}


module Synapse.ML.Training.Losses
    ( -- * @LossFn@ type alias

      LossFn

      -- * Regression losses
    
    , mse
    , msle
    , mae
    , mape
    , logcosh
    ) where


import Synapse.LinearAlgebra ((+.), (*.), (/.), (**.))

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M

import Synapse.Autograd (Symbol(unSymbol), Symbolic)


-- | @LossFn@ type alias represents functions that are able to provide a reference of what relation between matrices needs to be minimised.
type LossFn a = Symbol (Mat a) -> Symbol (Mat a) -> Symbol (Mat a)


-- Regression losses

-- | Divides every element of a @Mat@ by a number of elements in said @Mat@.
mean :: (Symbolic a, Fractional a) => Symbol (Mat a) -> Symbol (Mat a)
mean mat = mat /. fromIntegral (M.nElements $ unSymbol mat)


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