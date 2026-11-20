{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

-- | Type-level tensor dimensions and their runtime witnesses.
module Synapse.Tensor.Shapes.Dimensions
    ( Shape (..)
    , ShapeSize
    ) where

import GHC.TypeLits (Nat, type (*))

import Synapse.Tensor.Shapes.Backend (BackendShape, backendDimensions)

{- | Product of the dimensions in a shape.
@ShapeSize '[]@ is @1@; @ShapeSize '[2, 3]@ is @6@; any zero dimension makes the product zero.
-}
type family ShapeSize (sh :: [Nat]) :: Nat where
    ShapeSize '[] = 1
    ShapeSize (dim ': rest) = dim * ShapeSize rest

{- | A tensor shape with known runtime dimensions and an Accelerate shape representation.

Methods report dimensions in left-to-right order.
Instances are supplied for scalars and dimension lists whose sizes can be reflected at runtime.
The reported sizes and their product must fit in 'Int'.
-}
class BackendShape sh => Shape (sh :: [Nat]) where
    -- | Runtime dimensions for the type-level shape.
    shape
        :: [Int]

    -- | Total number of elements; a scalar has one element.
    shapeSize
        :: Int
    shapeSize = product (shape @sh)

    -- | Number of dimensions; a scalar has rank zero.
    shapeRank
        :: Int
    shapeRank = length (shape @sh)

instance Shape '[] where
    shape = []
instance
    BackendShape (dim ': rest)
    => Shape (dim ': rest)
    where
    shape = backendDimensions @(dim ': rest)
