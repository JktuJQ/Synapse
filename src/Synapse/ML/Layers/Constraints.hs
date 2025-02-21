{- | Allows to constraint values of layers parameters.

@Constraint@ type alias represents functions that are able to constrain the values of matrix.

Constraints should be applied for matrices the @updateParameters@ function.
-}


module Synapse.ML.Layers.Constraints
    ( -- * @Constraint@ type alias
      
      Constraint

      -- * Value constraints
    
    , nonNegative
    , clampMin
    , clampMax
    , clampMinMax

      -- * Matrix-specific constraints
    
    , centralize
    ) where


import Synapse.LinearAlgebra ((-.), (+.))

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M

import Data.Foldable (toList)
import Data.Ord (clamp)


-- | @Constraint@ type alias represents functions that are able to constrain the values of matrix.
type Constraint a = Mat a -> Mat a


-- Value constraints

-- | Ensures that matrix values are non-negative.
nonNegative :: (Num a, Ord a) => Constraint a
nonNegative = fmap (max 0)

-- | Ensures that matrix values are more or equal than given value.
clampMin :: Ord a => a -> Constraint a
clampMin minimal = fmap (max minimal)

-- | Ensures that matrix values are less or equal than given value.
clampMax :: Ord a => a -> Constraint a
clampMax maximal = fmap (min maximal)

-- | Ensures that matrix values are clamped between given values.
clampMinMax :: Ord a => (a, a) -> Constraint a
clampMinMax = fmap . clamp


-- Matrix-specific constraints

-- | Ensures that matrix values are centralized by mean around given value.
centralize :: Fractional a => a -> Constraint a
centralize center mat = mat -. (sum (toList mat) / fromIntegral (M.nElements mat)) +. center 