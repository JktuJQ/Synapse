{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Tensor variables, accumulation evidence, and conversion of parameter records.
-- Generic field traversal is shared with backend execution through "Synapse.Tensor.Fields".
module Synapse.Autograd.Execution.Variable
  ( Unconstrained,
    Variables,
    ValueTree (..),
    Values (..),
    GradientView (..),
    add,
    ones,
    pack,
    constants,
    stopped,
  )
where

import Data.Bifunctor (bimap)
import Data.Kind (Constraint, Type)
import qualified Numeric.Backprop.Explicit as BPE
import qualified Synapse.Autograd.Tensor.Type as Tensor
import Synapse.Autograd.Variable (Context, Var (..), stopVariable)
import qualified Synapse.Autograd.Variable as AG
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import Synapse.Tensor.Type (WithTensor)
import qualified Synapse.Tensor.Type as T

-- | No additional requirements on tensor leaves.
class Unconstrained a

instance Unconstrained a

-- | Differentiable wrappers for an already flattened tensor product.
type family Variables s a = (result :: Type) | result -> s a where
  Variables s (T.Tensor sh a) = Tensor.Tensor s sh a
  Variables s (a, b) = (Variables s a, Variables s b)

-- | Numeric evidence for a tensor leaf or the two branches of a product.
-- User records are flattened before this description is constructed.
data ValueTree (c :: Type -> Constraint) a where
  TensorValues :: (Num (T.Tensor sh a), c (T.Tensor sh a)) => ValueTree c (T.Tensor sh a)
  PairValues :: ValueTree c a -> ValueTree c b -> ValueTree c (a, b)

-- | Accumulation requirements for tensor products.
class (ValueConstraints Unconstrained a) => Values a where
  type ValueConstraints (c :: Type -> Constraint) a :: Constraint
  values :: (ValueConstraints c a) => ValueTree c a

instance (Num (T.Tensor sh a)) => Values (T.Tensor sh a) where
  type ValueConstraints c (T.Tensor sh a) = c (T.Tensor sh a)
  values = TensorValues

instance (Values a, Values b) => Values (a, b) where
  type ValueConstraints c (a, b) = (ValueConstraints c a, ValueConstraints c b)
  values = PairValues (values @a) (values @b)

-- | Convert variable field products without changing their histories.
class (Values (Leaves T.Tensor a)) => GradientView s a where
  gradientFields :: WithTensor (Tensor.Tensor s) a -> Variables s (Leaves T.Tensor a)
  gradientRecord :: Variables s (Leaves T.Tensor a) -> WithTensor (Tensor.Tensor s) a

instance
  ( Values (Leaves T.Tensor a),
    Flatten (Tensor.Tensor s) (WithTensor (Tensor.Tensor s) a),
    Leaves (Tensor.Tensor s) (WithTensor (Tensor.Tensor s) a) ~ Variables s (Leaves T.Tensor a)
  ) =>
  GradientView s a
  where
  gradientFields = flatten @(Tensor.Tensor s)
  gradientRecord = rebuild @(Tensor.Tensor s)

-- | Combine gradient contributions with each tensor's accumulation rule.
add :: ValueTree Unconstrained a -> a -> a -> a
add TensorValues = AG.add
add (PairValues left right) = \(x1, x2) (y1, y2) -> (add left x1 y1, add right x2 y2)

-- | Construct a tensor-shaped seed for every output leaf.
ones :: ValueTree Unconstrained a -> a -> a
ones TensorValues = AG.one
ones (PairValues left right) = bimap (ones left) (ones right)

-- | Join output variables without registering their containing record as an input.
pack :: forall s a. (Context s) => ValueTree Unconstrained a -> Variables s a -> BPE.BVar s a
pack TensorValues = unVar . Tensor.unTensor
pack (PairValues left right) = \(x, y) ->
  BPE.liftOp2
    (BPE.AF (add left))
    (BPE.AF (add right))
    (BPE.op2 (\a b -> ((a, b), id)))
    (pack @s left x)
    (pack @s right y)

-- | Introduce each tensor leaf without registering it as a differentiation input.
constants :: forall s a. ValueTree Unconstrained a -> a -> Variables s a
constants TensorValues = Tensor.Tensor . Var . BPE.auto
constants (PairValues left right) = bimap (constants @s left) (constants @s right)

-- | Preserve tensor values while removing all incoming tape edges.
stopped :: forall s a. ValueTree Unconstrained a -> Variables s a -> Variables s a
stopped TensorValues = Tensor.Tensor . stopVariable . Tensor.unTensor
stopped (PairValues left right) = bimap (stopped @s left) (stopped @s right)
