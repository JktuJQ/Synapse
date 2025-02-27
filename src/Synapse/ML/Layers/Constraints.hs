{- | Allows to constraint values of layers parameters.

@ConstraintFn@ type alias represents functions that are able to constrain the values of matrix
and @Constraint@ newtype wraps @ConstraintFn@s.

@ConstraintFn@s should be applied on matrices from the @updateParameters@ function.
-}


module Synapse.ML.Layers.Constraints
    ( -- * @ConstraintFn@ type alias and @Constraint@ newtype
      
      ConstraintFn

    , Constraint (Constraint, unConstraint)
    , applyConstraints

      -- * Value constraints
    
    , nonNegative
    , clampMin
    , clampMax
    , clampMinMax

      -- * Matrix-specific constraints
    
    , centralize
    ) where


import Synapse.LinearAlgebra (ElementwiseScalarOps((+.), (-.)), SingletonOps(unSingleton, mean))

import Synapse.LinearAlgebra.Mat (Mat)

import Data.Foldable (foldl')
import Data.Ord (clamp)


-- | @ConstraintFn@ type alias represents functions that are able to constrain the values of matrix.
type ConstraintFn a = Mat a -> Mat a

-- | @Constraint@ newtype wraps @ConstraintFn@s - functions that are able to constrain the values of matrix.
newtype Constraint a = Constraint
    { unConstraint :: ConstraintFn a  -- ^ Unwraps @Constraint@ newtype.
    }

-- | Applies all constraints on a matrix.
applyConstraints :: [Constraint a] -> Mat a -> Mat a
applyConstraints constraints mat = foldl' (\mat' (Constraint fn) -> fn mat') mat constraints


-- Value constraints

-- | Ensures that matrix values are non-negative.
nonNegative :: (Num a, Ord a) => ConstraintFn a
nonNegative = fmap (max 0)

-- | Ensures that matrix values are more or equal than given value.
clampMin :: Ord a => a -> ConstraintFn a
clampMin minimal = fmap (max minimal)

-- | Ensures that matrix values are less or equal than given value.
clampMax :: Ord a => a -> ConstraintFn a
clampMax maximal = fmap (min maximal)

-- | Ensures that matrix values are clamped between given values.
clampMinMax :: Ord a => (a, a) -> ConstraintFn a
clampMinMax = fmap . clamp


-- Matrix-specific constraints

-- | Ensures that matrix values are centralized by mean around given value.
centralize :: Fractional a => a -> ConstraintFn a
centralize center mat = mat -. unSingleton (mean mat) +. center
