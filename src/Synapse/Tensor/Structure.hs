{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoDeepSubsumption #-}

-- | Tensor structures, product decomposition, and fieldwise transformations.
module Synapse.Tensor.Structure
    ( -- * Tensor structures
      TensorStructure (..)

      -- * Tensor products
    , ProductFields
    , ValueTree (..)
    ) where

import Data.Bifunctor (bimap)
import Data.Kind (Constraint, Type)
import GHC.Generics (Generic (..), K1 (..), M1 (..), (:*:) (..))

import Synapse.Tensor.Representation (Tensor, TensorLike)

-- | Collect the canonical tensor products of generic fields.
type family GenericFields f where
    GenericFields (K1 i a) = TensorProduct Tensor a
    GenericFields (M1 i metadata f) = GenericFields f
    GenericFields (f :*: g) = (GenericFields f, GenericFields g)

-- | Convert a generic representation using the original fields as shape and type witnesses.
class GenericProduct t original actual where
    genericFlatten
        :: actual x
        -> ProductFields t (GenericFields original)
    genericRebuild
        :: ProductFields t (GenericFields original)
        -> actual x

instance (TensorStructure a, b ~ WithTensor t a) => GenericProduct t (K1 i a) (K1 j b) where
    genericFlatten (K1 x) = flatten @a @t x
    genericRebuild = K1 . rebuild @a @t
instance GenericProduct t f g => GenericProduct t (M1 i m f) (M1 j n g) where
    genericFlatten (M1 x) = genericFlatten @t @f x
    genericRebuild = M1 . genericRebuild @t @f
instance (GenericProduct t f h, GenericProduct t g i) => GenericProduct t (f :*: g) (h :*: i) where
    genericFlatten (a :*: b) = (genericFlatten @t @f a, genericFlatten @t @g b)
    genericRebuild (a, b) = genericRebuild @t @f a :*: genericRebuild @t @g b

class TensorProduct t p ~ ProductFields t (TensorProduct Tensor p) => ProductView t p
instance TensorProduct t p ~ ProductFields t (TensorProduct Tensor p) => ProductView t p

class AnyTensor a
instance AnyTensor a
class ProductFields Tensor fields ~ fields => ProductType c fields where
    productTree
        :: ValueTree c fields
instance c (Tensor sh a) => ProductType c (Tensor sh a) where
    productTree = TensorValues
instance (ProductType c a, ProductType c b) => ProductType c (a, b) where
    productTree = PairValues (productTree @c @a) (productTree @c @b)

mapProduct
    :: ValueTree c fields
    -> (forall sh a. t sh a -> u sh a)
    -> ProductFields t fields
    -> ProductFields u fields
mapProduct TensorValues f = f
mapProduct (PairValues l r) f = bimap (mapProduct l f) (mapProduct r f)

-- | Replace tensor representations in a nested binary product.
type family ProductFields (t :: TensorLike) fields = result | result -> t fields where
    ProductFields t (Tensor sh a) = t sh a
    ProductFields t (a, b) = (ProductFields t a, ProductFields t b)

-- | Evidence that every tensor in a nested product satisfies @c@.
data ValueTree (c :: Type -> Constraint) a where
    TensorValues :: c (Tensor sh a) => ValueTree c (Tensor sh a)
    PairValues :: ValueTree c a -> ValueTree c b -> ValueTree c (a, b)

{- | A tensor or a nonempty product of tensor fields.
Records derive 'Generic' and define their representation correspondence.

@
instance TensorStructure (Parameters Tensor) where
    type WithTensor t (Parameters Tensor) = Parameters t
@

Field constraints and fieldwise combination use generic defaults.
All fields must be tensors or nested tensor structures;
sums and ordinary configuration fields are not supported.
-}
class
    (WithTensor Tensor p ~ p, ProductType AnyTensor (TensorProduct Tensor p), forall t. ProductView t p) =>
    TensorStructure p
    where
    -- | Replace tensor representations while preserving the surrounding structure.
    type WithTensor (t :: TensorLike) p = (r :: Type) | r -> t p

    -- | Tensor fields collected into nested pairs; nesting follows the generic product representation.
    type TensorProduct (t :: TensorLike) p :: Type

    type TensorProduct t p = ProductFields t (GenericFields (Rep p))

    -- | Require @c@ on each tensor field.
    type TensorConstraints (c :: Type -> Constraint) p :: Constraint

    type TensorConstraints c p = ProductType c (TensorProduct Tensor p)

    -- | Collect tensor fields into nested products, preserving their values and field order.
    flatten
        :: forall t
         . WithTensor t p
        -> TensorProduct t p
    default flatten
        :: forall t
         . ( Generic p
           , Generic (WithTensor t p)
           , GenericProduct t (Rep p) (Rep (WithTensor t p))
           , TensorProduct t p ~ ProductFields t (GenericFields (Rep p))
           )
        => WithTensor t p
        -> TensorProduct t p
    flatten = genericFlatten @t @(Rep p) . from

    -- | Restore a structure from its tensor product; inverse of 'flatten'.
    rebuild
        :: forall t
         . TensorProduct t p
        -> WithTensor t p
    default rebuild
        :: forall t
         . ( Generic p
           , Generic (WithTensor t p)
           , GenericProduct t (Rep p) (Rep (WithTensor t p))
           , TensorProduct t p ~ ProductFields t (GenericFields (Rep p))
           )
        => TensorProduct t p
        -> WithTensor t p
    rebuild = to . genericRebuild @t @(Rep p)

    -- | Apply a representation change to every tensor field.
    mapTensorStructure
        :: forall t u
         . (forall sh a. t sh a -> u sh a)
        -> WithTensor t p
        -> WithTensor u p
    mapTensorStructure f =
        rebuild @p @u
            . mapProduct (productTree @AnyTensor @(TensorProduct Tensor p)) f
            . flatten @p @t

    -- | Combine corresponding tensor fields, passing the left field before the right field.
    zipWithTensorStructure
        :: forall c
         . TensorConstraints c p
        => (forall a. c a => a -> a -> a)
        -> p
        -> p
        -> p
    zipWithTensorStructure f left right =
        rebuild @p @Tensor $
            visit (tensorConstraints @p @c) (flatten @p @Tensor left) (flatten @p @Tensor right)
      where
        visit
            :: ValueTree c a
            -> a
            -> a
            -> a
        visit TensorValues a b = f a b
        visit (PairValues l r) (a, b) (x, y) = (visit l a x, visit r b y)

    -- | Supply constraint evidence for every tensor in the canonical product.
    tensorConstraints
        :: forall c
         . TensorConstraints c p
        => ValueTree c (TensorProduct Tensor p)
    default tensorConstraints
        :: forall c
         . (TensorConstraints c p, TensorConstraints c p ~ ProductType c (TensorProduct Tensor p))
        => ValueTree c (TensorProduct Tensor p)
    tensorConstraints = productTree @c @(TensorProduct Tensor p)

instance TensorStructure (Tensor sh a) where
    type WithTensor t (Tensor sh a) = t sh a
    type TensorProduct t (Tensor sh a) = t sh a
    flatten = id
    rebuild = id
instance (TensorStructure a1, TensorStructure a2) => TensorStructure (a1, a2) where
    type WithTensor t (a1, a2) = (WithTensor t a1, WithTensor t a2)
instance (TensorStructure a1, TensorStructure a2, TensorStructure a3) => TensorStructure (a1, a2, a3) where
    type WithTensor t (a1, a2, a3) = (WithTensor t a1, WithTensor t a2, WithTensor t a3)
instance
    (TensorStructure a1, TensorStructure a2, TensorStructure a3, TensorStructure a4)
    => TensorStructure (a1, a2, a3, a4)
    where
    type WithTensor t (a1, a2, a3, a4) = (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4)
instance
    (TensorStructure a1, TensorStructure a2, TensorStructure a3, TensorStructure a4, TensorStructure a5)
    => TensorStructure (a1, a2, a3, a4, a5)
    where
    type
        WithTensor t (a1, a2, a3, a4, a5) =
            (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4, WithTensor t a5)
instance
    ( TensorStructure a1
    , TensorStructure a2
    , TensorStructure a3
    , TensorStructure a4
    , TensorStructure a5
    , TensorStructure a6
    )
    => TensorStructure (a1, a2, a3, a4, a5, a6)
    where
    type
        WithTensor t (a1, a2, a3, a4, a5, a6) =
            (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4, WithTensor t a5, WithTensor t a6)
instance
    ( TensorStructure a1
    , TensorStructure a2
    , TensorStructure a3
    , TensorStructure a4
    , TensorStructure a5
    , TensorStructure a6
    , TensorStructure a7
    )
    => TensorStructure (a1, a2, a3, a4, a5, a6, a7)
    where
    type
        WithTensor t (a1, a2, a3, a4, a5, a6, a7) =
            ( WithTensor t a1
            , WithTensor t a2
            , WithTensor t a3
            , WithTensor t a4
            , WithTensor t a5
            , WithTensor t a6
            , WithTensor t a7
            )
instance
    ( TensorStructure a1
    , TensorStructure a2
    , TensorStructure a3
    , TensorStructure a4
    , TensorStructure a5
    , TensorStructure a6
    , TensorStructure a7
    , TensorStructure a8
    )
    => TensorStructure (a1, a2, a3, a4, a5, a6, a7, a8)
    where
    type
        WithTensor t (a1, a2, a3, a4, a5, a6, a7, a8) =
            ( WithTensor t a1
            , WithTensor t a2
            , WithTensor t a3
            , WithTensor t a4
            , WithTensor t a5
            , WithTensor t a6
            , WithTensor t a7
            , WithTensor t a8
            )
instance
    ( TensorStructure a1
    , TensorStructure a2
    , TensorStructure a3
    , TensorStructure a4
    , TensorStructure a5
    , TensorStructure a6
    , TensorStructure a7
    , TensorStructure a8
    , TensorStructure a9
    )
    => TensorStructure (a1, a2, a3, a4, a5, a6, a7, a8, a9)
    where
    type
        WithTensor t (a1, a2, a3, a4, a5, a6, a7, a8, a9) =
            ( WithTensor t a1
            , WithTensor t a2
            , WithTensor t a3
            , WithTensor t a4
            , WithTensor t a5
            , WithTensor t a6
            , WithTensor t a7
            , WithTensor t a8
            , WithTensor t a9
            )
