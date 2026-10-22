{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Extensible correspondence between ordinary and differentiable values.
module Synapse.Autograd.Differentiable
  ( Differentiable,
  )
where

import Data.Kind (Type)
import Synapse.Autograd.Type (Var)

-- | The representation of a value inside a differentiation scope.
-- Scalar types can use 'Var' directly.
-- Parameterized containers generally need a newtype around their variable representation to support partially applied constructors in operation classes.
-- For example, a container parameterized by shape and element type can provide the following adapter and instance:
--
-- @
-- newtype ContainerVar s sh a = ContainerVar (Var s (Container sh a))
-- type instance Differentiable s (Container sh a) = ContainerVar s sh a
-- @
--
-- Pairs preserve the separate representations of their components.
type family Differentiable s a :: Type

type instance Differentiable s Int = Var s Int

type instance Differentiable s Integer = Var s Integer

type instance Differentiable s Float = Var s Float

type instance Differentiable s Double = Var s Double

type instance Differentiable s (a, b) = (Differentiable s a, Differentiable s b)
