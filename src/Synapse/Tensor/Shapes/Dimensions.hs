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
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

-- | Tensor dimensions at the type and value levels.
module Synapse.Tensor.Shapes.Dimensions
  ( ShapeToDIM,
    Shape (..),
    ShapeSize,
    Reverse,
    backendShape,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, Nat, natVal, type (*))

-- | Tail-recursive worker for 'Reverse'.
type family ReverseAcc (xs :: [Nat]) (acc :: [Nat]) :: [Nat] where
  ReverseAcc '[] acc = acc
  ReverseAcc (x ': xs) acc = ReverseAcc xs (x ': acc)

-- | Reverse a type-level list.
--
-- Synapse exposes shapes left-to-right, e.g. @'[batch, features]@. Accelerate
-- shapes are built by appending axes on the right, e.g. @Z :. batch :. features@.
-- Reversing first lets the recursive builder preserve the user-facing order.
type family Reverse (xs :: [Nat]) :: [Nat] where
  Reverse xs = ReverseAcc xs '[]

-- | Type-level product of all dimensions in a shape.
type family ShapeSize (sh :: [Nat]) :: Nat where
  ShapeSize '[] = 1
  ShapeSize (dim ': rest) = dim * ShapeSize rest

-- | Recursive builder for Accelerate shape types from a reversed Synapse shape.
type family ShapeToDIMRev (sh :: [Nat]) :: Type where
  ShapeToDIMRev '[] = A.Z
  ShapeToDIMRev (dim ': rest) = ShapeToDIMRev rest A.:. Int

-- | Convert a Synapse shape to Accelerate's recursive shape type.
--
-- The Synapse shape @'[d1, d2, d3]@ maps to Accelerate's @Z :. Int :. Int :. Int@,
-- preserving the same left-to-right axis order at the value level.
type ShapeToDIM (sh :: [Nat]) = ShapeToDIMRev (Reverse sh)

-- | Convert a type-level natural number to an 'Int'.
dimVal :: forall n. (KnownNat n) => Int
dimVal = fromInteger . toInteger $ natVal (Proxy @n)

-- | Runtime witness builder for reversed Synapse shapes.
class ShapeRev (sh :: [Nat]) where
  -- | Build the Accelerate shape value for a reversed Synapse shape.
  backendShapeRev :: proxy sh -> ShapeToDIMRev sh

  -- | Build the user-facing dimension list for a reversed Synapse shape.
  shapeRev :: proxy sh -> [Int]

instance ShapeRev '[] where
  backendShapeRev _ = A.Z
  shapeRev _ = []

instance (KnownNat dim, ShapeRev rest) => ShapeRev (dim ': rest) where
  backendShapeRev _ = backendShapeRev (Proxy @rest) A.:. dimVal @dim
  shapeRev _ = shapeRev (Proxy @rest) <> [dimVal @dim]

-- | A shape whose dimensions can be materialized at runtime.
-- All methods describe the same dimensions in left-to-right order.
class (A.Shape (ShapeToDIM sh), ShapeRev (Reverse sh)) => Shape (sh :: [Nat]) where
  -- | Runtime dimensions for the type-level shape.
  shape :: [Int]

  -- | Total number of elements; a scalar has one element.
  shapeSize :: Int
  shapeSize = product (shape @sh)


  -- | Number of dimensions; a scalar has rank zero.
  shapeRank :: Int
  shapeRank = length (shape @sh)

instance Shape '[] where
  shape = []

instance
  (A.Shape (ShapeToDIM (dim ': rest)), ShapeRev (Reverse (dim ': rest))) =>
  Shape (dim ': rest)
  where
  shape = shapeRev (Proxy @(Reverse (dim ': rest)))

-- | Internal representation of dimensions used by tensor implementations.
-- This function is not re-exported by the public tensor or shapes modules.
backendShape :: forall sh proxy. (Shape sh) => proxy sh -> ShapeToDIM sh
backendShape _ = backendShapeRev (Proxy @(Reverse sh))
