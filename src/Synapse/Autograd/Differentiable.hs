{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Differentiable values and descriptions of their independently registered fields.
-- Records with one constructor and at least one field can use the @Generic@ default.
-- Derive @Generic@ for both representations and specify the correspondence between their fields:
--
-- @
-- {-# LANGUAGE DeriveGeneric #-}
-- {-# LANGUAGE TypeFamilies #-}
-- {-# LANGUAGE UndecidableInstances #-}
--
-- import GHC.Generics (Generic)
-- import Synapse.Autograd.Differentiable
--
-- data Parameters a b = Parameters { first :: a, second :: b }
--   deriving Generic
--
-- instance (Differentiable a, Differentiable b) => Differentiable (Parameters a b) where
--   type WithGrad s (Parameters a b) = Parameters (WithGrad s a) (WithGrad s b)
-- @
--
-- Instead of deriving 'Generic', the same instance can implement 'structure' explicitly:
--
-- @
-- structure = record Parameters Parameters
--   <+> field first first
--   <+> field second second
-- @
--
-- Each selector pair must identify the same field in the ordinary and differentiable records.
-- Fields must appear in constructor argument order and must preserve their values and variable histories.
-- Constructors must reconstruct both representations from their declared fields without transforming the field values.
-- Each leaf is registered independently, so the containing record needs no 'Backpropagatable' instance.
-- Nested records are traversed recursively.
-- The generic default supports products, not sum types or constructors without fields.
module Synapse.Autograd.Differentiable
  ( Differentiable (..),
    Structure,
    leaf,
    Record,
    Field,
    record,
    field,
    (<+>),
  )
where

import Data.Kind (Type)
import GHC.Generics (Generic (..), K1 (..), M1 (..), (:*:) (..))
import Synapse.Autograd.Differentiable.Structure (Field, Structure)
import qualified Synapse.Autograd.Differentiable.Structure as S
import Synapse.Autograd.Type (Backpropagatable, Var)

type instance S.Representation s a = WithGrad s a

-- | Describe a value registered as one input, together with its variable wrapper.
-- The value's 'Backpropagatable' instance initializes and combines its gradient contributions.
-- Wrapping and unwrapping must be inverse operations that preserve the variable and its differentiation history.
leaf ::
  (Backpropagatable a) =>
  (forall s. Var s a -> WithGrad s a) ->
  (forall s. WithGrad s a -> Var s a) ->
  Structure a
leaf = S.Leaf

-- | The result type of a curried constructor.
type family Result constructor where
  Result (a -> rest) = Result rest
  Result result = result

-- | The remaining argument types of a curried constructor.
type family Arguments constructor where
  Arguments (a -> rest) = a ': Arguments rest
  Arguments result = '[]

-- | A record description awaiting its remaining fields.
-- Add fields with '<+>' until the description becomes a 'Structure'.
newtype Record a fields = Record (S.Fields a fields -> Structure a)

-- | The result after adding a field to a record description.
type family RecordResult a fields where
  RecordResult a '[] = Structure a
  RecordResult a (field ': rest) = Record a (field ': rest)

-- | Complete the description as soon as every constructor argument has a field.
class Complete fields where
  complete :: Record a fields -> RecordResult a fields

instance Complete '[] where
  complete (Record build) = build S.NoFields

instance Complete (a ': rest) where
  complete = id

-- | Start a record description with its ordinary and differentiable constructors.
-- Constructors must accept corresponding fields in the same order and the record must contain at least one field.
-- Add one 'field' for each constructor argument using '<+>'.
record ::
  (rest ~ S.Constructor (Arguments rest) (Result rest)) =>
  (a -> rest) ->
  (forall s. WithGrad s a -> S.GradientConstructor s (Arguments rest) (Result rest)) ->
  Record (Result rest) (a ': Arguments rest)
record constructor differentiableConstructor = Record (\descriptions -> S.Record descriptions constructor differentiableConstructor)

-- | Select the same component from the ordinary and differentiable records.
-- Selectors must preserve the component's value and differentiation history.
-- The component's 'Differentiable' instance describes its independently registered leaves.
field ::
  (Differentiable a) =>
  (record -> a) ->
  (forall s. WithGrad s record -> WithGrad s a) ->
  Field record a
field = S.Field structure

infixl 4 <+>

-- | Add the next field in constructor argument order.
-- Adding the final field produces a 'Structure' without an explicit finishing operation.
(<+>) :: (Complete rest) => Record a (b ': rest) -> Field a b -> RecordResult a rest
Record build <+> next = complete (Record (build . S.MoreFields next))

-- | The fields of a single-constructor generic representation.
type family Fields (f :: Type -> Type) :: Type where
  Fields (K1 i a) = a
  Fields (M1 i metadata f) = Fields f
  Fields (f :*: g) = (Fields f, Fields g)

-- | Convert a generic representation to its nested field product.
class GenericFields f where
  fields :: f p -> Fields f
  rebuildFields :: Fields f -> f p

instance GenericFields (K1 i a) where
  fields (K1 value) = value
  rebuildFields = K1

instance (GenericFields f) => GenericFields (M1 i metadata f) where
  fields (M1 value) = fields value
  rebuildFields = M1 . rebuildFields

instance (GenericFields f, GenericFields g) => GenericFields (f :*: g) where
  fields (left :*: right) = (fields left, fields right)
  rebuildFields (left, right) = rebuildFields left :*: rebuildFields right

-- | Match the generic fields of the ordinary and differentiable records.
class GenericVariables s a where
  variableFields :: WithGrad s a -> WithGrad s (Fields (Rep a))
  variableRecord :: WithGrad s (Fields (Rep a)) -> WithGrad s a

instance
  ( Generic (WithGrad s a),
    GenericFields (Rep (WithGrad s a)),
    Fields (Rep (WithGrad s a)) ~ WithGrad s (Fields (Rep a))
  ) =>
  GenericVariables s a
  where
  variableFields = fields . from
  variableRecord = to . rebuildFields

-- | Values supported by reverse-mode differentiation.
-- Records with one constructor can derive 'structure' through 'Generic' when their differentiable fields correspond to 'WithGrad' of their ordinary fields.
-- Each 'WithGrad' representation must uniquely identify its ordinary type and differentiation scope.
-- An instance describes representations and input registration; derivative rules belong to the operations on those representations.
class Differentiable a where
  -- | The representation of a value inside differentiation scope @s@.
  -- Custom types define this correspondence explicitly; there is no automatic fallback to 'Var'.
  -- A distinct variable newtype can preserve the correspondence required by the injectivity annotation.
  type WithGrad s a = (result :: Type) | result -> s a

  -- | Describe the independently registered fields of a value.
  -- Variable wrappers can use 'leaf'; records can use 'record', 'field', and '<+>'.
  structure :: Structure a
  default structure ::
    ( Generic a,
      GenericFields (Rep a),
      Differentiable (Fields (Rep a)),
      forall s. GenericVariables s a
    ) =>
    Structure a
  structure =
    S.Record
      (S.MoreFields (S.Field (structure @(Fields (Rep a))) (fields . from) variableFields) S.NoFields)
      (to . rebuildFields)
      variableRecord

instance Differentiable Int where
  type WithGrad s Int = Var s Int
  structure = leaf id id

instance Differentiable Integer where
  type WithGrad s Integer = Var s Integer
  structure = leaf id id

instance Differentiable Float where
  type WithGrad s Float = Var s Float
  structure = leaf id id

instance Differentiable Double where
  type WithGrad s Double = Var s Double
  structure = leaf id id

instance (Differentiable a, Differentiable b) => Differentiable (a, b) where
  type WithGrad s (a, b) = (WithGrad s a, WithGrad s b)
  structure =
    record (,) (,)
      <+> field fst fst
      <+> field snd snd

instance (Differentiable a, Differentiable b, Differentiable c) => Differentiable (a, b, c) where
  type WithGrad s (a, b, c) = (WithGrad s a, WithGrad s b, WithGrad s c)

instance (Differentiable a, Differentiable b, Differentiable c, Differentiable d) => Differentiable (a, b, c, d) where
  type WithGrad s (a, b, c, d) = (WithGrad s a, WithGrad s b, WithGrad s c, WithGrad s d)

instance (Differentiable a, Differentiable b, Differentiable c, Differentiable d, Differentiable e) => Differentiable (a, b, c, d, e) where
  type WithGrad s (a, b, c, d, e) = (WithGrad s a, WithGrad s b, WithGrad s c, WithGrad s d, WithGrad s e)

instance (Differentiable a, Differentiable b, Differentiable c, Differentiable d, Differentiable e, Differentiable f) => Differentiable (a, b, c, d, e, f) where
  type WithGrad s (a, b, c, d, e, f) = (WithGrad s a, WithGrad s b, WithGrad s c, WithGrad s d, WithGrad s e, WithGrad s f)

instance (Differentiable a, Differentiable b, Differentiable c, Differentiable d, Differentiable e, Differentiable f, Differentiable g) => Differentiable (a, b, c, d, e, f, g) where
  type WithGrad s (a, b, c, d, e, f, g) = (WithGrad s a, WithGrad s b, WithGrad s c, WithGrad s d, WithGrad s e, WithGrad s f, WithGrad s g)

instance (Differentiable a, Differentiable b, Differentiable c, Differentiable d, Differentiable e, Differentiable f, Differentiable g, Differentiable h) => Differentiable (a, b, c, d, e, f, g, h) where
  type WithGrad s (a, b, c, d, e, f, g, h) = (WithGrad s a, WithGrad s b, WithGrad s c, WithGrad s d, WithGrad s e, WithGrad s f, WithGrad s g, WithGrad s h)

instance (Differentiable a, Differentiable b, Differentiable c, Differentiable d, Differentiable e, Differentiable f, Differentiable g, Differentiable h, Differentiable i) => Differentiable (a, b, c, d, e, f, g, h, i) where
  type WithGrad s (a, b, c, d, e, f, g, h, i) = (WithGrad s a, WithGrad s b, WithGrad s c, WithGrad s d, WithGrad s e, WithGrad s f, WithGrad s g, WithGrad s h, WithGrad s i)
