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
    backendCoordinates,
    strides,
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
-- Synapse exposes shapes left-to-right as @'[batch, features]@.
-- Backend shapes append axes on the right as @Z :. batch :. features@.
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
-- The Synapse shape @'[d1, d2, d3]@ maps to Accelerate's @Z :. Int :. Int :. Int@, preserving the same left-to-right axis order at the value level.
type ShapeToDIM (sh :: [Nat]) = ShapeToDIMRev (Reverse sh)

-- | Convert a type-level natural number to an 'Int'.
dimVal :: forall n. (KnownNat n) => Int
dimVal = fromInteger . toInteger $ natVal (Proxy @n)

-- | Runtime witness builder for reversed Synapse shapes.
class (A.Shape (ShapeToDIMRev sh)) => ShapeRev (sh :: [Nat]) where
  -- | Build the Accelerate shape value for a reversed Synapse shape.
  backendShapeRev :: proxy sh -> ShapeToDIMRev sh

  -- | Prepend dimensions in user-facing order to a suffix.
  -- The accumulator avoids repeatedly appending lists while traversing the reversed shape.
  shapeRev :: proxy sh -> [Int] -> [Int]

  -- | Prepend coordinate expressions in user-facing order to a suffix, extracting them directly from the recursive index representation.
  coordinatesRev :: A.Exp (ShapeToDIMRev sh) -> [A.Exp Int] -> [A.Exp Int]

instance ShapeRev '[] where
  backendShapeRev _ = A.Z
  shapeRev _ suffix = suffix
  coordinatesRev _ suffix = suffix

instance (KnownNat dim, ShapeRev rest) => ShapeRev (dim ': rest) where
  backendShapeRev _ = backendShapeRev (Proxy @rest) A.:. dimVal @dim
  shapeRev _ suffix = shapeRev (Proxy @rest) (dimVal @dim : suffix)
  coordinatesRev ix suffix = coordinatesRev @rest (A.indexTail ix) (A.indexHead ix : suffix)

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
  shape = shapeRev (Proxy @(Reverse (dim ': rest))) []

-- | Internal representation of dimensions used by tensor implementations.
-- This function is not re-exported by the public tensor or shapes modules.
backendShape :: forall sh proxy. (Shape sh) => proxy sh -> ShapeToDIM sh
backendShape _ = backendShapeRev (Proxy @(Reverse sh))

-- | Internal extraction of coordinates from the backend index representation.
-- The result follows the same left-to-right axis order as 'shape'.
backendCoordinates :: forall sh. (Shape sh) => A.Exp (ShapeToDIM sh) -> [A.Exp Int]
backendCoordinates ix = coordinatesRev @(Reverse sh) ix []

-- | Row-major strides.
-- The rightmost axis has stride one; a scalar has none.
strides :: [Int] -> [Int]
strides = drop 1 . scanr (*) 1
