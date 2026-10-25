{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Descriptions of independently differentiated leaves and record fields.
module Synapse.Autograd.Differentiable.Structure
  ( Unconstrained,
    Representation,
    Constructor,
    GradientConstructor,
    Structure (..),
    Field (..),
    Fields (..),
    rebuild,
  )
where

import Data.Kind (Constraint, Type)
import Synapse.Autograd.Type (Backpropagatable, Var)

-- | No additional requirements on the registered leaves.
class Unconstrained a

instance Unconstrained a

-- | The scope-indexed representation selected by a structural description.
type family Representation s a :: Type

-- | A constructor accepting the declared fields in order.
type family Constructor (fields :: [Type]) result where
  Constructor '[] result = result
  Constructor (a ': rest) result = a -> Constructor rest result

-- | The corresponding constructor for differentiable fields.
type family GradientConstructor s (fields :: [Type]) result where
  GradientConstructor s '[] result = Representation s result
  GradientConstructor s (a ': rest) result = Representation s a -> GradientConstructor s rest result

-- | A leaf variable or a record of independently differentiated fields.
data Structure (c :: Type -> Constraint) a where
  Leaf ::
    (Backpropagatable a, c a) =>
    (forall s. Var s a -> Representation s a) ->
    (forall s. Representation s a -> Var s a) ->
    Structure c a
  Record ::
    Fields c a fields ->
    Constructor fields a ->
    (forall s. GradientConstructor s fields a) ->
    Structure c a

-- | A field description and its selectors in the two representations.
data Field c record a
  = Field
      (Structure c a)
      (record -> a)
      (forall s. Representation s record -> Representation s a)

-- | Fields in constructor argument order.
data Fields c record fields where
  NoFields :: Fields c record '[]
  MoreFields :: Field c record a -> Fields c record rest -> Fields c record (a ': rest)

-- | Reconstruct an ordinary value from its declared fields.
rebuild :: (forall b. Field c a b -> b) -> Fields c a fields -> Constructor fields a -> a
rebuild _ NoFields value = value
rebuild select (MoreFields child rest) constructor = rebuild select rest (constructor (select child))
