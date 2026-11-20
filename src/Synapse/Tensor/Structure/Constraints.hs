{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

{- | Constraints and evidence for individual tensor fields and nested products.

The constraint families apply a requirement to every tensor in a product.
'ProductConstraintsWitness' constructs a matching tree of evidence for traversals and differentiation.
-}
module Synapse.Tensor.Structure.Constraints
    ( FieldEvidence (..)
    , ElementEvidence (..)
    , RequirementEvidence (..)
    , ProductRequirements
    , ProductFieldConstraints
    , ProductElementConstraints
    , ProductConstraintsWitness (..)
    ) where

import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy (..))

import Synapse.Tensor.Representation (DType, Tensor)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Structure.Product (ProductFields, TensorField, ValueTree (..))

-- | Evidence that a canonical tensor field satisfies @c@.
data FieldEvidence c field where
    FieldEvidence
        :: c field => FieldEvidence c field

-- | Evidence for the shape, dtype, and element constraint of a tensor field.
data ElementEvidence c field where
    ElementEvidence
        :: (Shape sh, DType a, c a) => ElementEvidence c (Tensor sh a)

-- | Evidence for a requirement on representations, shape, and element type.
data RequirementEvidence requirements representations field where
    RequirementEvidence
        :: requirements representations sh a => RequirementEvidence requirements representations (Tensor sh a)

-- | Apply a requirement to every tensor field in a product.
type family ProductRequirements requirements representations fields :: Constraint where
    ProductRequirements requirements representations (TensorField (Tensor sh a)) =
        requirements representations sh a
    ProductRequirements requirements representations (left, right) =
        ( ProductRequirements requirements representations left
        , ProductRequirements requirements representations right
        )

-- | Apply a constraint to every complete tensor type in a product.
type family ProductFieldConstraints (c :: Type -> Constraint) fields :: Constraint where
    ProductFieldConstraints c (TensorField (Tensor sh a)) = c (Tensor sh a)
    ProductFieldConstraints c (left, right) =
        ( ProductFieldConstraints c left
        , ProductFieldConstraints c right
        )

-- | Require known shapes, supported dtypes, and an element constraint for every field.
type family ProductElementConstraints (c :: Type -> Constraint) fields :: Constraint where
    ProductElementConstraints c (TensorField (Tensor sh a)) = (Shape sh, DType a, c a)
    ProductElementConstraints c (left, right) =
        (ProductElementConstraints c left, ProductElementConstraints c right)

-- | Construct a product shape witness and constraint evidence for each tensor field.
class ProductFields Tensor fields ~ fields => ProductConstraintsWitness fields where
    -- | Witness the product shape without requiring constraints on its fields.
    productTree
        :: ValueTree Proxy fields

    -- | Construct evidence for a representation, shape, and element requirement at each field.
    requirementTree
        :: forall requirements representations
         . ProductRequirements requirements representations fields
        => ValueTree (RequirementEvidence requirements representations) fields

    -- | Construct evidence for a constraint on each complete tensor type.
    fieldTree
        :: forall c
         . ProductFieldConstraints c fields
        => ValueTree (FieldEvidence c) fields

    -- | Construct evidence for shape, dtype, and element constraints.
    elementTree
        :: forall c
         . ProductElementConstraints c fields
        => ValueTree (ElementEvidence c) fields

instance ProductConstraintsWitness (TensorField (Tensor sh a)) where
    productTree = TensorValues Proxy
    requirementTree = TensorValues RequirementEvidence
    fieldTree = TensorValues FieldEvidence
    elementTree = TensorValues ElementEvidence
instance
    (ProductConstraintsWitness left, ProductConstraintsWitness right)
    => ProductConstraintsWitness (left, right)
    where
    productTree = PairValues (productTree @left) (productTree @right)
    requirementTree = PairValues (requirementTree @left) (requirementTree @right)
    fieldTree = PairValues (fieldTree @left) (fieldTree @right)
    elementTree = PairValues (elementTree @left) (elementTree @right)
