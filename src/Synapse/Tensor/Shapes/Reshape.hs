{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Type-level compatibility of tensor reshapes.
module Synapse.Tensor.Shapes.Reshape
    ( CanReshape
    )
where

import Data.Kind (Constraint)
import GHC.TypeLits (ErrorMessage (ShowType, Text, (:<>:)), Nat, TypeError)

import Synapse.Tensor.Shapes.Dimensions (ShapeSize)

-- | Compare element counts and report both shapes and sizes when they differ.
type family CheckReshape (from :: [Nat]) (to :: [Nat]) (fromSize :: Nat) (toSize :: Nat) :: Constraint where
    CheckReshape from to size size = ()
    CheckReshape from to fromSize toSize =
        TypeError
            ( 'Text "Cannot reshape tensor from shape "
                ':<>: 'ShowType from
                ':<>: 'Text " with "
                ':<>: 'ShowType fromSize
                ':<>: 'Text " element(s) to shape "
                ':<>: 'ShowType to
                ':<>: 'Text " with "
                ':<>: 'ShowType toSize
                ':<>: 'Text " element(s)"
            )

{- | Require equal element counts in the source and target shapes.

For example, @CanReshape '[2, 3] '[6]@ holds.
A scalar has one element and may be reshaped to any shape whose dimension product is one.
Empty shapes such as @'[0, 3]@ and @'[2, 0]@ have equal element counts and may be reshaped into each other.
Here the scalar shape is @'[]@; a shape containing a zero dimension describes an empty tensor.
-}
type CanReshape from to =
    CheckReshape from to (ShapeSize from) (ShapeSize to)
