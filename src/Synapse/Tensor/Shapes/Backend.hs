{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Conversion between type-level dimensions and Accelerate shapes and indices.
module Synapse.Tensor.Shapes.Backend
    ( -- * Shape representation
      ShapeToDIM
    , BackendShape

      -- * Dimension lists
    , Reverse
    , strides
    , backendDimensions

      -- * Shape and index conversion
    , backendShape
    , backendCoordinates
    , backendIndex
    ) where

import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, Nat, natVal)

import qualified Data.Array.Accelerate as A

-- | Reflect a type-level natural number as an Int; the dimension must fit in that type.
dimVal
    :: forall n
     . KnownNat n
    => Int
dimVal = fromInteger . toInteger $ natVal (Proxy @n)

-- | Compute row-major strides for a dimension list.
strides
    :: [Int]
    -> [Int]
strides = drop 1 . scanr (*) 1

-- Dimension list reversal

-- | Reverse a dimension list onto an accumulator.
type family ReverseAcc (xs :: [Nat]) (acc :: [Nat]) :: [Nat] where
    ReverseAcc '[] acc = acc
    ReverseAcc (x ': xs) acc = ReverseAcc xs (x ': acc)

{- | Reverse a type-level list of dimensions.
The backend builders consume reversed lists so that recursive shape construction preserves axis order.
For example, @Reverse '[2, 3, 4]@ is @'[4, 3, 2]@.
-}
type family Reverse (xs :: [Nat]) :: [Nat] where
    Reverse xs = ReverseAcc xs '[]

-- Backend shape representation

-- | Build an Accelerate index type from a reversed dimension list; dimension sizes are not part of this type.
type family ShapeToDIMRev (sh :: [Nat]) :: Type where
    ShapeToDIMRev '[] = A.Z
    ShapeToDIMRev (dim ': rest) = ShapeToDIMRev rest A.:. Int

{- | Accelerate index type corresponding to the rank of a Synapse shape.

@ShapeToDIM '[2, 3]@ is @A.Z A.:. Int A.:. Int@; @ShapeToDIM '[]@ is @A.Z@.
The type records rank only; 'backendShape' supplies the dimension sizes in left-to-right order.
-}
type ShapeToDIM (sh :: [Nat]) = ShapeToDIMRev (Reverse sh)

{- | Build runtime shapes and index expressions from reversed dimension lists.
Accumulator arguments restore left-to-right order when producing lists of dimensions or coordinates.
-}
class A.Shape (ShapeToDIMRev sh) => ShapeRev (sh :: [Nat]) where
    -- | Build an array shape from the reversed dimension list.
    backendShapeRev
        :: proxy sh
        -> ShapeToDIMRev sh

    -- | Prepend dimensions in left-to-right order to a suffix.
    shapeRev
        :: proxy sh
        -> [Int]
        -> [Int]

    -- | Extract coordinates from the recursive index and prepend them in left-to-right order to a suffix.
    coordinatesRev
        :: A.Exp (ShapeToDIMRev sh)
        -> [A.Exp Int]
        -> [A.Exp Int]

    -- | Build an index from coordinates supplied in reversed axis order.
    indexRev
        :: [A.Exp Int]
        -> A.Exp (ShapeToDIMRev sh)

instance ShapeRev '[] where
    backendShapeRev _ = A.Z
    shapeRev _ suffix = suffix
    coordinatesRev _ suffix = suffix
    indexRev [] = A.index0
    indexRev _ = error "Synapse.Tensor.backendIndex: inconsistent coordinate rank"
instance (KnownNat dim, ShapeRev rest) => ShapeRev (dim ': rest) where
    backendShapeRev _ = backendShapeRev (Proxy @rest) A.:. dimVal @dim
    shapeRev _ suffix = shapeRev (Proxy @rest) (dimVal @dim : suffix)
    coordinatesRev ix suffix = coordinatesRev @rest (A.indexTail ix) (A.indexHead ix : suffix)
    indexRev (coord : coords) = A.lift (indexRev @rest coords A.:. coord)
    indexRev [] = error "Synapse.Tensor.backendIndex: inconsistent coordinate rank"

-- | Dimensions with a known Accelerate shape representation.
type BackendShape sh = (A.Shape (ShapeToDIM sh), ShapeRev (Reverse sh))

-- | Reflect dimensions in left-to-right order.
backendDimensions
    :: forall sh
     . BackendShape sh
    => [Int]
backendDimensions = shapeRev (Proxy @(Reverse sh)) []

-- Backend shape and index conversion

-- | Reflect the dimensions of @sh@ as an array shape.
backendShape
    :: forall sh proxy
     . BackendShape sh
    => proxy sh
    -> ShapeToDIM sh
backendShape _ = backendShapeRev (Proxy @(Reverse sh))

-- | Extract index coordinates in left-to-right axis order.
backendCoordinates
    :: forall sh
     . BackendShape sh
    => A.Exp (ShapeToDIM sh)
    -> [A.Exp Int]
backendCoordinates ix = coordinatesRev @(Reverse sh) ix []

-- | Construct an index from coordinates in left-to-right axis order.
backendIndex
    :: forall sh
     . BackendShape sh
    => [A.Exp Int]
    -> A.Exp (ShapeToDIM sh)
backendIndex = indexRev @(Reverse sh) . reverse
