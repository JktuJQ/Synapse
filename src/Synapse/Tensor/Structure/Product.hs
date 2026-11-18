{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Named tensor products and their traversals.
module Synapse.Tensor.Structure.Product
    ( FieldPath
    , TensorField (..)
    , ProductFields
    , ProductValues
    , ValueTree (..)
    , prefixFields
    , foldProduct
    , traverseProduct
    , zipProduct
    ) where

import Data.Kind (Type)

import Synapse.Tensor.Representation (Tensor, TensorLike)

-- | Replace tensor representations in a nested binary product.
type family ProductFields (t :: TensorLike) fields = result | result -> t fields where
    ProductFields t (TensorField (Tensor sh a)) = TensorField (t sh a)
    ProductFields t (a, b) = (ProductFields t a, ProductFields t b)

{- | Names of enclosing record fields or zero-based tuple positions.
A standalone tensor has path @[]@. Components are literal strings.
-}
type FieldPath = [String]

-- | A tensor value and its structural field path.
data TensorField field = TensorField
    { tensorFieldPath :: FieldPath
    , tensorFieldValue :: field
    }
    deriving (Eq, Show)

-- | Tensor values without the field paths of a product.
type family ProductValues fields where
    ProductValues (TensorField (Tensor sh a)) = Tensor sh a
    ProductValues (left, right) = (ProductValues left, ProductValues right)

-- | A tensor product with a value indexed by the tensor type at each leaf.
data ValueTree (evidence :: Type -> Type) fields where
    -- | Evidence for one tensor field.
    TensorValues
        :: evidence (Tensor sh a)
        -> ValueTree evidence (TensorField (Tensor sh a))
    -- | Evidence for both sides of a product.
    PairValues
        :: ValueTree evidence left
        -> ValueTree evidence right
        -> ValueTree evidence (left, right)

-- | Prefix every path in a product using its structure witness.
prefixFields
    :: forall t evidence fields
     . ValueTree evidence fields
    -> String
    -> ProductFields t fields
    -> ProductFields t fields
prefixFields (TensorValues _) name (TensorField path value) = TensorField (name : path) value
prefixFields (PairValues left right) name (x, y) =
    (prefixFields @t left name x, prefixFields @t right name y)

-- | Fold a product from left to right using evidence for every field.
foldProduct
    :: forall t evidence fields result
     . ValueTree evidence fields
    -> (forall sh a. evidence (Tensor sh a) -> FieldPath -> result -> t sh a -> result)
    -> result
    -> ProductFields t fields
    -> result
foldProduct (TensorValues evidence) step initial (TensorField path tensor) =
    step evidence path initial tensor
foldProduct (PairValues left right) step initial (x, y) =
    foldProduct @t right step (foldProduct @t left step initial x) y

-- | Transform a product with effects using evidence for every field.
traverseProduct
    :: forall t1 t2 evidence fields f
     . Applicative f
    => ValueTree evidence fields
    -> (forall sh a. evidence (Tensor sh a) -> FieldPath -> t1 sh a -> f (t2 sh a))
    -> ProductFields t1 fields
    -> f (ProductFields t2 fields)
traverseProduct (TensorValues evidence) transform (TensorField path tensor) =
    TensorField path <$> transform evidence path tensor
traverseProduct (PairValues left right) transform (x, y) =
    (,) <$> traverseProduct @t1 @t2 left transform x <*> traverseProduct @t1 @t2 right transform y

-- | Combine corresponding product fields using evidence for every field.
zipProduct
    :: forall t1 t2 t3 evidence fields
     . ValueTree evidence fields
    -> (forall sh a. evidence (Tensor sh a) -> FieldPath -> t1 sh a -> t2 sh a -> t3 sh a)
    -> ProductFields t1 fields
    -> ProductFields t2 fields
    -> ProductFields t3 fields
zipProduct (TensorValues evidence) combine (TensorField path x) (TensorField _ y) =
    TensorField path (combine evidence path x y)
zipProduct (PairValues left right) combine (x1, x2) (y1, y2) =
    (zipProduct @t1 @t2 @t3 left combine x1 y1, zipProduct @t1 @t2 @t3 right combine x2 y2)
