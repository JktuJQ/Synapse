{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Product decomposition of tensor structures for execution and differentiation.
module Synapse.Tensor.Structure
    ( -- * Product decomposition
      Leaves
    , Flatten (..)
    ) where

import GHC.Generics (Generic (..), K1 (..), M1 (..), (:*:) (..))

import Synapse.Tensor.Representation (TensorLike)

{- | Tensor leaves of a structure, collected into nested binary products.

A tensor of representation @t@ is retained unchanged.
Other types are decomposed through their 'Generic' representation, recursively visiting their fields.
Constructor and selector metadata are discarded; field order is preserved.
The nesting follows the generic product representation rather than a fixed left- or right-associated layout.
-}
type family Leaves (t :: TensorLike) a where
    Leaves t (t sh a) = t sh a
    Leaves t a = GenericLeaves t (Rep a)

{- | Convert a tensor structure to and from its binary field product.

Tensor instances preserve the leaf; the generic instance recursively converts single-constructor records.
Generic records must be nonempty, with every field holding a tensor or a nested tensor structure.
Sum types and ordinary configuration fields are not supported by this decomposition.
Both conversions must preserve tensor values and be mutual inverses.
-}
class Flatten (t :: TensorLike) a where
    -- | Decompose the surrounding structure without reshaping or evaluating its tensors.
    flatten
        :: a
        -> Leaves t a

    -- | Restore the surrounding structure from its tensor leaves.
    rebuild
        :: Leaves t a
        -> a

instance {-# OVERLAPPING #-} Flatten t (t sh a) where
    flatten = id
    rebuild = id
instance
    {-# OVERLAPPABLE #-}
    (Generic a, GenericFlatten t (Rep a), Leaves t a ~ GenericLeaves t (Rep a))
    => Flatten t a
    where
    flatten = genericFlatten @t . from
    rebuild = to . genericRebuild @t

{- | Tensor leaves of a generic product representation.
-
Metadata wrappers are removed, fields are decomposed through 'Leaves', and products become pairs.
Empty constructors and sums have no corresponding equation.
-}
type family GenericLeaves t f where
    GenericLeaves t (K1 i a) = Leaves t a
    GenericLeaves t (M1 i metadata f) = GenericLeaves t f
    GenericLeaves t (f :*: g) = (GenericLeaves t f, GenericLeaves t g)

-- | Decompose and reconstruct generic product fields without changing tensor values.
class GenericFlatten t f where
    -- | Remove metadata wrappers and collect tensor leaves in field order.
    genericFlatten
        :: f p
        -> GenericLeaves t f

    -- | Rebuild generic fields and metadata wrappers from the leaf product.
    genericRebuild
        :: GenericLeaves t f
        -> f p

instance Flatten t a => GenericFlatten t (K1 i a) where
    genericFlatten (K1 value) = flatten @t value
    genericRebuild = K1 . rebuild @t
instance GenericFlatten t f => GenericFlatten t (M1 i metadata f) where
    genericFlatten (M1 value) = genericFlatten @t value
    genericRebuild = M1 . genericRebuild @t
instance (GenericFlatten t f, GenericFlatten t g) => GenericFlatten t (f :*: g) where
    genericFlatten (left :*: right) = (genericFlatten @t left, genericFlatten @t right)
    genericRebuild (left, right) = genericRebuild @t left :*: genericRebuild @t right
