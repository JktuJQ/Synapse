{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Reconstruction, gradient accumulation, and variable conversion for structured values.
module Synapse.Autograd.Execution.Traversal
  ( add,
    ones,
    pack,
    constants,
    stopped,
  )
where

import Data.Functor.Identity (Identity (..))
import Data.Vinyl.Core (Rec (..))
import qualified Numeric.Backprop.Explicit as BPE
import Synapse.Autograd.Differentiable.Structure (Constructor, Field (..), Fields (..), GradientConstructor, Representation, Structure (..))
import Synapse.Autograd.Type (Context, Var (..), stopVariable)
import qualified Synapse.Autograd.Type as AG

-- | Rebuild an ordinary record by selecting each declared field.
rebuild :: (forall b. Field a b -> b) -> Fields a fields -> Constructor fields a -> a
rebuild _ NoFields value = value
rebuild select (MoreFields child rest) constructor = rebuild select rest (constructor (select child))

-- | Rebuild a differentiable record by selecting each declared variable representation.
rebuildVariables ::
  forall s a fields.
  (forall b. Field a b -> Representation s b) ->
  Fields a fields ->
  GradientConstructor s fields a ->
  Representation s a
rebuildVariables _ NoFields value = value
rebuildVariables select (MoreFields child rest) constructor = rebuildVariables @s select rest (constructor (select child))

-- | Combine gradient contributions by applying each leaf's accumulation rule.
add :: Structure a -> a -> a -> a
add (Leaf _ _) = AG.add
add (Record fields constructor _) = \x y ->
  rebuild (\(Field child getter _) -> add child (getter x) (getter y)) fields constructor

-- | Construct the default output seed with the same structure as the value.
ones :: Structure a -> a -> a
ones (Leaf _ _) = AG.one
ones (Record fields constructor _) = \value ->
  rebuild (\(Field child getter _) -> ones child (getter value)) fields constructor

-- | Apply a curried constructor to an ordinary record of arguments.
construct :: Constructor fields a -> Rec Identity fields -> a
construct value RNil = value
construct constructor (Identity value :& rest) = construct (constructor value) rest

-- | Project ordinary field values in constructor argument order.
project :: Fields a fields -> a -> Rec Identity fields
project NoFields _ = RNil
project (MoreFields (Field _ getter _) rest) value = Identity (getter value) :& project rest value

-- | Gradient accumulation functions for the declared fields.
additions :: Fields a fields -> Rec BPE.AddFunc fields
additions NoFields = RNil
additions (MoreFields (Field child _ _) rest) = BPE.AF (add child) :& additions rest

-- | Join each field's representation into its own output variable.
variables :: forall s a fields. (Context s) => Fields a fields -> Representation s a -> Rec (BPE.BVar s) fields
variables NoFields _ = RNil
variables (MoreFields (Field child _ getter) rest) value = pack @s child (getter @s value) :& variables @s rest value

-- | Join output variables through the public isomorphism operation of <https://hackage.haskell.org/package/backprop backprop>.
pack :: forall s a. (Context s) => Structure a -> Representation s a -> BPE.BVar s a
pack (Leaf _ unwrap) = unVar . unwrap @s
pack (Record fields constructor _) =
  BPE.isoVarN (additions fields) (construct constructor) (project fields) . variables @s fields

-- | Introduce values as constants rather than differentiable inputs.
constants :: forall s a. Structure a -> a -> Representation s a
constants (Leaf wrap _) = wrap @s . Var . BPE.auto
constants (Record fields _ constructor) = \value ->
  rebuildVariables @s (\(Field child getter _) -> constants @s child (getter value)) fields (constructor @s)

-- | Preserve primal values while removing all incoming tape edges.
stopped :: forall s a. Structure a -> Representation s a -> Representation s a
stopped (Leaf wrap unwrap) = wrap @s . stopVariable . unwrap @s
stopped (Record fields _ constructor) = \value ->
  rebuildVariables @s (\(Field child _ getter) -> stopped @s child (getter @s value)) fields (constructor @s)
