{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Descriptions of independently differentiated leaves and record fields.
module Synapse.Autograd.Differentiable.Structure
  ( Representation,
    Constructor,
    GradientConstructor,
    Structure (..),
    Field (..),
    Fields (..),
  )
where

import Data.Kind (Type)
import Synapse.Autograd.Type (Backpropagatable, Var)

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
data Structure a where
  Leaf ::
    (Backpropagatable a) =>
    (forall s. Var s a -> Representation s a) ->
    (forall s. Representation s a -> Var s a) ->
    Structure a
  Record ::
    Fields a fields ->
    Constructor fields a ->
    (forall s. GradientConstructor s fields a) ->
    Structure a

-- | A field description and its selectors in the two representations.
data Field record a
  = Field
      (Structure a)
      (record -> a)
      (forall s. Representation s record -> Representation s a)

-- | Fields in constructor argument order.
data Fields record fields where
  NoFields :: Fields record '[]
  MoreFields :: Field record a -> Fields record rest -> Fields record (a ': rest)
