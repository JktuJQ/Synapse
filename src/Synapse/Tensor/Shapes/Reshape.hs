{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Type-level compatibility checks for reshaping tensors.
module Synapse.Tensor.Shapes.Reshape
  ( CanReshape,
  )
where

import Data.Kind (Constraint)
import GHC.TypeLits (ErrorMessage (ShowType, Text, (:<>:)), Nat, TypeError)
import Synapse.Tensor.Shapes.Dimensions (ShapeSize)

-- | Constraint proving that two shapes have the same number of elements.
type CanReshape from to =
  CheckReshape from to (ShapeSize from) (ShapeSize to)

-- | Implementation of 'CanReshape' with a readable type error.
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
