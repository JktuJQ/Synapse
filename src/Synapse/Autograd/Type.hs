{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Differentiation contexts, variables, and supported tensor structures.
-- Each tensor leaf is registered independently and receives a gradient with its original shape.
-- Records derive @Generic@ and define 'WithTensor'; no additional class instance is required.
-- Ordinary numbers and element expressions are not differentiable leaves.
-- Represent an individual differentiable number as a scalar tensor.
module Synapse.Autograd.Type
  ( Context,
    Var (..),
    Differentiable,
    WithGrad,
    ParametersConstraints,
  )
where

import Synapse.Autograd.Execution.Variable (GradientView, Values (..))
import Synapse.Autograd.Tensor.Type (Tensor)
import Synapse.Autograd.Variable (Context, Var (..))
import Synapse.Tensor.Fields (Flatten, Leaves)
import Synapse.Tensor.Type (WithTensor)
import qualified Synapse.Tensor.Type as T

-- | The tensor representation of a value inside differentiation scope @s@.
type WithGrad s a = WithTensor (Tensor s) a

-- | Require a constraint on every tensor leaf of a parameter structure.
-- The containing records need no numeric instances.
type ParametersConstraints c a = ValueConstraints c (Leaves T.Tensor a)

-- | Tensor structures supported by reverse-mode differentiation.
-- Single-constructor records are traversed recursively through their generic fields.
class
  (Flatten T.Tensor a, Values (Leaves T.Tensor a), forall s. GradientView s a) =>
  Differentiable a

instance
  (Flatten T.Tensor a, Values (Leaves T.Tensor a), forall s. GradientView s a) =>
  Differentiable a
