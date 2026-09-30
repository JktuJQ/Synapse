{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Type-level tensor shapes.
--
-- Synapse represents tensor shapes as type-level lists of natural numbers. This
-- module provides runtime helpers for those type-level shapes.
module Synapse.Tensor.Shape
  ( ShapeToDIM,
    ShapeSize,
    KnownShape,
    CanBroadcast,
    CanReshape,
    shapeVal,
    shapeList,
    shape,
    shapeSize,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits
  ( ErrorMessage (ShowType, Text, (:<>:)),
    KnownNat,
    Nat,
    TypeError,
    natVal,
    type (*),
  )

-- | Reverse a type-level list.
--
-- Synapse exposes shapes left-to-right, e.g. @'[batch, features]@. Accelerate
-- shapes are built by appending axes on the right, e.g. @Z :. batch :. features@.
-- Reversing first lets the recursive builder preserve the user-facing order.
type family Reverse (xs :: [Nat]) :: [Nat] where
  Reverse xs = ReverseAcc xs '[]

-- | Tail-recursive worker for 'Reverse'.
type family ReverseAcc (xs :: [Nat]) (acc :: [Nat]) :: [Nat] where
  ReverseAcc '[] acc = acc
  ReverseAcc (x ': xs) acc = ReverseAcc xs (x ': acc)

-- | Recursive builder for Accelerate shape types from a reversed Synapse shape.
type family ShapeToDIMRev (sh :: [Nat]) :: Type where
  ShapeToDIMRev '[] = A.Z
  ShapeToDIMRev (dim ': rest) = ShapeToDIMRev rest A.:. Int

-- | Convert a Synapse shape to Accelerate's recursive shape type.
--
-- The Synapse shape @'[d1, d2, d3]@ maps to Accelerate's @Z :. Int :. Int :. Int@,
-- preserving the same left-to-right axis order at the value level.
type ShapeToDIM (sh :: [Nat]) = ShapeToDIMRev (Reverse sh)

-- | Runtime witness builder for reversed Synapse shapes.
class KnownShapeRev (sh :: [Nat]) where
  -- | Build the Accelerate shape value for a reversed Synapse shape.
  shapeValRev :: proxy sh -> ShapeToDIMRev sh

  -- | Build the user-facing dimension list for a reversed Synapse shape.
  shapeListRev :: proxy sh -> [Int]

instance KnownShapeRev '[] where
  shapeValRev _ = A.Z
  shapeListRev _ = []

instance (KnownNat dim, KnownShapeRev rest) => KnownShapeRev (dim ': rest) where
  shapeValRev _ = shapeValRev (Proxy @rest) A.:. dimVal @dim
  shapeListRev _ = shapeListRev (Proxy @rest) <> [dimVal @dim]

-- | Constraint proving that a shape can be materialized at runtime.
type KnownShape (sh :: [Nat]) =
  (A.Shape (ShapeToDIM sh), KnownShapeRev (Reverse sh))

-- | Convert the type-level shape to the internal backend shape value.
shapeVal :: forall sh proxy. (KnownShape sh) => proxy sh -> ShapeToDIM sh
shapeVal _ = shapeValRev (Proxy @(Reverse sh))

-- | Convert the type-level shape to a list of runtime dimensions.
shapeList :: forall sh proxy. (KnownShape sh) => proxy sh -> [Int]
shapeList _ = shapeListRev (Proxy @(Reverse sh))

-- | Runtime dimensions for a type-level shape.
shape :: forall sh. (KnownShape sh) => [Int]
shape = shapeList (Proxy @sh)

-- | Total number of elements in a type-level shape.
shapeSize :: forall sh. (KnownShape sh) => Int
shapeSize = product (shape @sh)

-- | Convert a type-level natural number to an 'Int'.
dimVal :: forall n. (KnownNat n) => Int
dimVal = fromInteger . toInteger $ natVal (Proxy @n)

-- | Type-level product of all dimensions in a shape.
type family ShapeSize (sh :: [Nat]) :: Nat where
  ShapeSize '[] = 1
  ShapeSize (dim ': rest) = dim * ShapeSize rest

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

-- | Check whether a single source dimension can broadcast to a target dimension.
type family CanBroadcastDim (fromDim :: Nat) (toDim :: Nat) :: Constraint where
  CanBroadcastDim 1 toDim = ()
  CanBroadcastDim dim dim = ()
  CanBroadcastDim fromDim toDim =
    TypeError
      ( 'Text "Cannot broadcast dimension "
          ':<>: 'ShowType fromDim
          ':<>: 'Text " to "
          ':<>: 'ShowType toDim
      )

-- | Compare reversed shapes dimension-by-dimension for broadcast compatibility.
type family BroadcastSuffix (from :: [Nat]) (to :: [Nat]) :: Constraint where
  BroadcastSuffix '[] _ = ()
  BroadcastSuffix (fromDim ': fromRest) '[] =
    TypeError
      ( 'Text "Cannot broadcast source shape with extra leading dimension "
          ':<>: 'ShowType fromDim
      )
  BroadcastSuffix (fromDim ': fromRest) (toDim ': toRest) =
    (CanBroadcastDim fromDim toDim, BroadcastSuffix fromRest toRest)

-- | Constraint proving that one shape can be broadcast to another.
--
-- Broadcast compatibility is checked from trailing dimensions, matching NumPy
-- and PyTorch semantics.
type CanBroadcast from to = BroadcastSuffix (Reverse from) (Reverse to)
