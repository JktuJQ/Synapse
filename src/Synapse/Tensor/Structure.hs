{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

{- | Tensor structures, product decomposition, and fieldwise transformations.

Import this module qualified to distinguish its fieldwise operations from tensor element operations.
General traversals use a caller-defined requirement on representations, shapes, and element types.
They accept any tensor representation, including materialized data.
Products returned by 'flatten' store a t'TensorField' at each leaf, pairing its path with its value.
Use a single representation for 'fold',
a promoted pair for 'map' and 'traverse',
and a promoted triple for 'zipWith'.
The @Fields@ variants accept ordinary constraints such as 'Num' on complete canonical tensors.
The @Elements@ variants constrain element types and also supply 'Shape' and 'DType' evidence.
All callbacks receive the field path first.
All folds and effectful traversals visit fields from left to right.
-}
module Synapse.Tensor.Structure
    ( -- * Tensor structures
      TensorStructure (..)
    , FieldPath
    , TensorField (..)
    , ProductFields

      -- * Structure manipulations
    , map
    , fold
    , traverse
    , zipWith
    , mapFields
    , foldFields
    , traverseFields
    , zipWithFields
    , mapElements
    , foldElements
    , traverseElements
    , zipWithElements

      -- * Constraints
    , TensorConstraints
    , FieldConstraints
    , ElementConstraints
    ) where

import Data.Functor.Identity (Identity (..))
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))
import GHC.Generics (C, D, Generic (..), K1 (..), M1 (..), S, Selector (..), (:*:) (..))
import GHC.TypeLits (Nat)
import Prelude hiding (map, traverse, zipWith)

import Synapse.Tensor.Representation (DType, Tensor, TensorLike)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Structure.Constraints
    ( ElementEvidence (..)
    , FieldEvidence (..)
    , ProductConstraintsWitness (..)
    , ProductElementConstraints
    , ProductFieldConstraints
    , ProductRequirements
    , RequirementEvidence (..)
    )
import Synapse.Tensor.Structure.Product
    ( FieldPath
    , ProductFields
    , TensorField (..)
    , foldProduct
    , prefixFields
    , traverseProduct
    , zipProduct
    )

-- | Collect the canonical tensor products of generic fields.
type family GenericFields f where
    GenericFields (K1 i a) = TensorProduct Tensor a
    GenericFields (M1 i metadata f) = GenericFields f
    GenericFields (f :*: g) = (GenericFields f, GenericFields g)

-- | Convert generic products, attaching selectors or positional names to tensor leaves.
class GenericProduct t original actual where
    -- | Count immediate fields in the original generic representation.
    genericFieldCount
        :: proxy original
        -> Int

    -- | Flatten a representation, starting unnamed field positions at the given offset.
    genericFlatten
        :: Int
        -> actual x
        -> ProductFields t (GenericFields original)

    -- | Rebuild a generic representation from its product.
    genericRebuild
        :: ProductFields t (GenericFields original)
        -> actual x

instance (TensorStructure a, b ~ WithTensor t a) => GenericProduct t (K1 i a) (K1 j b) where
    genericFieldCount _ = 1
    genericFlatten _ (K1 x) = flatten @a @t x
    genericRebuild = K1 . rebuild @a @t
instance GenericProduct t f g => GenericProduct t (M1 D m f) (M1 D n g) where
    genericFieldCount _ = genericFieldCount @t @f @g (Proxy @f)
    genericFlatten offset (M1 x) = genericFlatten @t @f offset x
    genericRebuild = M1 . genericRebuild @t @f
instance GenericProduct t f g => GenericProduct t (M1 C m f) (M1 C n g) where
    genericFieldCount _ = genericFieldCount @t @f @g (Proxy @f)
    genericFlatten offset (M1 x) = genericFlatten @t @f offset x
    genericRebuild = M1 . genericRebuild @t @f
instance
    (Selector m, GenericProduct t f g, ProductConstraintsWitness (GenericFields f))
    => GenericProduct t (M1 S m f) (M1 S n g)
    where
    genericFieldCount _ = 1
    genericFlatten offset (M1 x) =
        let name = case selName (undefined :: M1 S m f ()) of
                "" -> show offset
                selector -> selector
         in prefixFields @t (productTree @(GenericFields f)) name (genericFlatten @t @f 0 x)
    genericRebuild = M1 . genericRebuild @t @f
instance (GenericProduct t f h, GenericProduct t g i) => GenericProduct t (f :*: g) (h :*: i) where
    genericFieldCount _ = genericFieldCount @t @f @h (Proxy @f) + genericFieldCount @t @g @i (Proxy @g)
    genericFlatten offset (a :*: b) =
        ( genericFlatten @t @f offset a
        , genericFlatten @t @g (offset + genericFieldCount @t @f @h (Proxy @f)) b
        )
    genericRebuild (a, b) = genericRebuild @t @f a :*: genericRebuild @t @g b

-- | Evidence that changing a structure's representation preserves its product shape.
class TensorProduct t p ~ ProductFields t (TensorProduct Tensor p) => ProductView t p

instance TensorProduct t p ~ ProductFields t (TensorProduct Tensor p) => ProductView t p

{- | A tensor or a nonempty product of tensor fields.
Records derive 'Generic' and define their representation correspondence.

@
instance TensorStructure (Parameters Tensor) where
    type WithTensor t (Parameters Tensor) = Parameters t
@

Product decomposition uses generic defaults.
All fields must be tensors or nested tensor structures;
sums and ordinary configuration fields are not supported.
-}
class
    ( WithTensor Tensor p ~ p
    , ProductConstraintsWitness (TensorProduct Tensor p)
    , forall t. ProductView t p
    ) =>
    TensorStructure p
    where
    -- | Replace tensor representations while preserving the surrounding structure.
    type WithTensor (t :: TensorLike) p = (r :: Type) | r -> t p

    -- | Named tensor fields collected into nested pairs following the generic product representation.
    type TensorProduct (t :: TensorLike) p :: Type

    type TensorProduct t p = ProductFields t (GenericFields (Rep p))

    {- | Collect named tensor fields into nested products in field order.
    Generic defaults prefix selector names or zero-based positions for unnamed fields.
    Paths must be unique and independent of tensor contents and representation.
    -}
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
    flatten = genericFlatten @t @(Rep p) 0 . from

    {- | Restore a structure from tensor values, ignoring their paths.
    Rebuilding a flattened structure preserves its values; flattening again restores canonical paths.
    -}
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

instance TensorStructure (Tensor sh a) where
    type WithTensor t (Tensor sh a) = t sh a
    type TensorProduct t (Tensor sh a) = TensorField (t sh a)
    flatten = TensorField []
    rebuild = tensorFieldValue
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
            ( WithTensor t a1
            , WithTensor t a2
            , WithTensor t a3
            , WithTensor t a4
            , WithTensor t a5
            , WithTensor t a6
            )
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

-- | Require exactly @requirements representations sh a@ for each tensor field.
type TensorConstraints (requirements :: k -> [Nat] -> Type -> Constraint) (representations :: k) p =
    ProductRequirements requirements representations (TensorProduct Tensor p)

-- | Require @c (Tensor sh a)@ for every canonical tensor field.
type FieldConstraints c p = ProductFieldConstraints c (TensorProduct Tensor p)

-- | Require a known shape, a supported dtype, and @c a@ for every tensor field.
type ElementConstraints c p = ProductElementConstraints c (TensorProduct Tensor p)

-- | Transform fields using a requirement on the input and output representations.
map
    :: forall requirements p t1 t2
     . (TensorStructure p, TensorConstraints requirements '(t1, t2) p)
    => (forall sh a. requirements '(t1, t2) sh a => FieldPath -> t1 sh a -> t2 sh a)
    -> WithTensor t1 p
    -> WithTensor t2 p
map transform = runIdentity . traverse @requirements @p @t1 @t2 (\path -> Identity . transform path)

-- | Fold fields from left to right, passing the accumulator before each field.
fold
    :: forall requirements p t result
     . (TensorStructure p, TensorConstraints requirements t p)
    => (forall sh a. requirements t sh a => FieldPath -> result -> t sh a -> result)
    -> result
    -> WithTensor t p
    -> result
fold step initial =
    foldProduct @t
        (requirementTree @(TensorProduct Tensor p) @requirements @t)
        (\RequirementEvidence -> step)
        initial
        . flatten @p @t

-- | Transform fields with effects, sequencing them from left to right.
traverse
    :: forall requirements p t1 t2 f
     . (TensorStructure p, Applicative f, TensorConstraints requirements '(t1, t2) p)
    => (forall sh a. requirements '(t1, t2) sh a => FieldPath -> t1 sh a -> f (t2 sh a))
    -> WithTensor t1 p
    -> f (WithTensor t2 p)
traverse transform =
    fmap (rebuild @p @t2)
        . traverseProduct @t1 @t2
            (requirementTree @(TensorProduct Tensor p) @requirements @'(t1, t2))
            (\RequirementEvidence -> transform)
        . flatten @p @t1

-- | Combine corresponding fields using a requirement on all three representations.
zipWith
    :: forall requirements p t1 t2 t3
     . (TensorStructure p, TensorConstraints requirements '(t1, t2, t3) p)
    => (forall sh a. requirements '(t1, t2, t3) sh a => FieldPath -> t1 sh a -> t2 sh a -> t3 sh a)
    -> WithTensor t1 p
    -> WithTensor t2 p
    -> WithTensor t3 p
zipWith combine left right =
    rebuild @p @t3 $
        zipProduct @t1 @t2 @t3
            (requirementTree @(TensorProduct Tensor p) @requirements @'(t1, t2, t3))
            (\RequirementEvidence -> combine)
            (flatten @p @t1 left)
            (flatten @p @t2 right)

-- | Transform each canonical tensor using a constraint on its complete type.
mapFields
    :: forall c p
     . (TensorStructure p, FieldConstraints c p)
    => (forall field. c field => FieldPath -> field -> field)
    -> p
    -> p
mapFields transform = runIdentity . traverseFields @c @p (\path -> Identity . transform path)

-- | Fold canonical tensor fields using a constraint on each complete field type.
foldFields
    :: forall c p result
     . (TensorStructure p, FieldConstraints c p)
    => (forall field. c field => FieldPath -> result -> field -> result)
    -> result
    -> p
    -> result
foldFields step initial =
    foldProduct @Tensor (fieldTree @(TensorProduct Tensor p) @c) (\FieldEvidence -> step) initial
        . flatten @p @Tensor

-- | Transform canonical tensor fields with effects and a constraint on each complete field type.
traverseFields
    :: forall c p f
     . (TensorStructure p, Applicative f, FieldConstraints c p)
    => (forall field. c field => FieldPath -> field -> f field)
    -> p
    -> f p
traverseFields transform =
    fmap (rebuild @p @Tensor)
        . traverseProduct @Tensor @Tensor
            (fieldTree @(TensorProduct Tensor p) @c)
            (\FieldEvidence -> transform)
        . flatten @p @Tensor

-- | Combine canonical tensor fields, passing the left field before the right field.
zipWithFields
    :: forall c p
     . (TensorStructure p, FieldConstraints c p)
    => (forall field. c field => FieldPath -> field -> field -> field)
    -> p
    -> p
    -> p
zipWithFields combine left right =
    rebuild @p @Tensor $
        zipProduct @Tensor @Tensor @Tensor
            (fieldTree @(TensorProduct Tensor p) @c)
            (\FieldEvidence -> combine)
            (flatten @p @Tensor left)
            (flatten @p @Tensor right)

-- | Transform fields with known shapes, supported dtypes, and an element constraint.
mapElements
    :: forall c p t1 t2
     . (TensorStructure p, ElementConstraints c p)
    => (forall sh a. (Shape sh, DType a, c a) => FieldPath -> t1 sh a -> t2 sh a)
    -> WithTensor t1 p
    -> WithTensor t2 p
mapElements transform = runIdentity . traverseElements @c @p @t1 @t2 (\path -> Identity . transform path)

-- | Fold fields from left to right using a constraint on each element type.
foldElements
    :: forall c p t result
     . (TensorStructure p, ElementConstraints c p)
    => (forall sh a. (Shape sh, DType a, c a) => FieldPath -> result -> t sh a -> result)
    -> result
    -> WithTensor t p
    -> result
foldElements step initial =
    foldProduct @t
        (elementTree @(TensorProduct Tensor p) @c)
        (\ElementEvidence -> step)
        initial
        . flatten @p @t

-- | Transform fields with effects and a constraint on each element type.
traverseElements
    :: forall c p t1 t2 f
     . (TensorStructure p, Applicative f, ElementConstraints c p)
    => (forall sh a. (Shape sh, DType a, c a) => FieldPath -> t1 sh a -> f (t2 sh a))
    -> WithTensor t1 p
    -> f (WithTensor t2 p)
traverseElements transform =
    fmap (rebuild @p @t2)
        . traverseProduct @t1 @t2
            (elementTree @(TensorProduct Tensor p) @c)
            (\ElementEvidence -> transform)
        . flatten @p @t1

-- | Combine corresponding fields with known shapes, supported dtypes, and an element constraint.
zipWithElements
    :: forall c p t1 t2 t3
     . (TensorStructure p, ElementConstraints c p)
    => (forall sh a. (Shape sh, DType a, c a) => FieldPath -> t1 sh a -> t2 sh a -> t3 sh a)
    -> WithTensor t1 p
    -> WithTensor t2 p
    -> WithTensor t3 p
zipWithElements combine left right =
    rebuild @p @t3 $
        zipProduct @t1 @t2 @t3
            (elementTree @(TensorProduct Tensor p) @c)
            (\ElementEvidence -> combine)
            (flatten @p @t1 left)
            (flatten @p @t2 right)
