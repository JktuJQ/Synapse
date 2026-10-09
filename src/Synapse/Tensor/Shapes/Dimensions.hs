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

-- | Tensor dimensions at the type and value levels.
module Synapse.Tensor.Shapes.Dimensions
  ( ShapeToDIM,
    Shape,
    ShapeSize,
    Reverse,
    shapeVal,
    shapeList,
    shape,
    shapeSize,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, Nat, natVal, type (*))

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
class ShapeRev (sh :: [Nat]) where
  -- | Build the Accelerate shape value for a reversed Synapse shape.
  shapeValRev :: proxy sh -> ShapeToDIMRev sh

  -- | Build the user-facing dimension list for a reversed Synapse shape.
  shapeListRev :: proxy sh -> [Int]

instance ShapeRev '[] where
  shapeValRev _ = A.Z
  shapeListRev _ = []

instance (KnownNat dim, ShapeRev rest) => ShapeRev (dim ': rest) where
  shapeValRev _ = shapeValRev (Proxy @rest) A.:. dimVal @dim
  shapeListRev _ = shapeListRev (Proxy @rest) <> [dimVal @dim]

-- | Constraint proving that a shape can be materialized at runtime.
type Shape (sh :: [Nat]) =
  (A.Shape (ShapeToDIM sh), ShapeRev (Reverse sh))

-- | Convert the type-level shape to the internal backend shape value.
shapeVal :: forall sh proxy. (Shape sh) => proxy sh -> ShapeToDIM sh
shapeVal _ = shapeValRev (Proxy @(Reverse sh))

-- | Convert the type-level shape to a list of runtime dimensions.
shapeList :: forall sh proxy. (Shape sh) => proxy sh -> [Int]
shapeList _ = shapeListRev (Proxy @(Reverse sh))

-- | Runtime dimensions for a type-level shape.
shape :: forall sh. (Shape sh) => [Int]
shape = shapeList (Proxy @sh)

-- | Total number of elements in a type-level shape.
shapeSize :: forall sh. (Shape sh) => Int
shapeSize = product (shape @sh)

-- | Convert a type-level natural number to an 'Int'.
dimVal :: forall n. (KnownNat n) => Int
dimVal = fromInteger . toInteger $ natVal (Proxy @n)

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

-- | Type-level product of all dimensions in a shape.
type family ShapeSize (sh :: [Nat]) :: Nat where
  ShapeSize '[] = 1
  ShapeSize (dim ': rest) = dim * ShapeSize rest
