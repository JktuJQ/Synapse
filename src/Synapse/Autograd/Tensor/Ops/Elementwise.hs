{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Nondifferentiable predicates and differentiable value selection.
module Synapse.Autograd.Tensor.Ops.Elementwise () where

import Synapse.Autograd.Tensor.Type (Tensor (..), TensorElement (..))
import Synapse.Autograd.Type (Backpropagatable, Context, Var, add, liftBinary, liftUnary, variableValue, zero)
import Synapse.Tensor.Construction (Scalar (..))
import qualified Synapse.Tensor.Construction as C
import Synapse.Tensor.DType (DType)
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Index as I
import Synapse.Tensor.Shapes.Dimensions (Shape)
import qualified Synapse.Tensor.Type as T

-- | Route the output gradient using the same mask as the forward selection.
-- Selection rather than multiplication keeps inactive infinite seeds from producing NaN.
selectVariable :: (Context s, Backpropagatable a, E.Conditional a) => E.Condition a -> Var s a -> Var s a -> Var s a
selectVariable mask = liftBinary add add $ \x y ->
  (E.select mask x y, \gradient -> (E.select mask gradient (zero x), E.select mask (zero y) gradient))

-- | Route a whole-computation gradient to the branch chosen by the scalar predicate.
condVariable :: (Context s, Backpropagatable a, E.Conditional a) => E.Predicate a -> Var s a -> Var s a -> Var s a
condVariable flag = liftBinary add add $ \x y ->
  (E.cond flag x y, \gradient -> (E.cond flag gradient (zero x), E.cond flag (zero y) gradient))

-- | Split equal extrema evenly and zero the contribution of the losing branch.
extremumVariable ::
  (Context s, Backpropagatable a, Fractional a, E.Ord a, E.Conditional a, E.Comparison a ~ E.Condition a) =>
  (a -> a -> a) ->
  (a -> a -> E.Comparison a) ->
  Var s a ->
  Var s a ->
  Var s a
extremumVariable operation loses = liftBinary add add $ \x y ->
  ( operation x y,
    \gradient ->
      let shared = E.select (x E.== y) (gradient / 2) gradient
       in (E.select (loses x y) (zero x) shared, E.select (loses y x) (zero y) shared)
  )

instance (Context s, Shape sh, DType a, Num a, Num (T.TensorElement a)) => E.Conditional (Tensor s sh a) where
  type Condition (Tensor s sh a) = T.Tensor sh Bool
  type Predicate (Tensor s sh a) = T.Tensor '[] Bool
  select mask (Tensor x) (Tensor y) = Tensor (selectVariable mask x y)
  cond flag (Tensor x) (Tensor y) = Tensor (condVariable flag x y)

instance (Context s, DType a, Num (T.TensorElement a)) => E.Conditional (TensorElement s a) where
  type Condition (TensorElement s a) = T.TensorElement Bool
  type Predicate (TensorElement s a) = T.TensorElement Bool
  select mask (TensorElement x) (TensorElement y) = TensorElement (selectVariable mask x y)
  cond flag (TensorElement x) (TensorElement y) = TensorElement (condVariable flag x y)

instance (E.Eq (T.Tensor sh a)) => E.Eq (Tensor s sh a) where
  type Comparison (Tensor s sh a) = T.Tensor sh Bool
  Tensor x == Tensor y = variableValue x E.== variableValue y
  Tensor x /= Tensor y = variableValue x E./= variableValue y

instance (E.Eq (T.TensorElement a)) => E.Eq (TensorElement s a) where
  type Comparison (TensorElement s a) = T.TensorElement Bool
  TensorElement x == TensorElement y = variableValue x E.== variableValue y
  TensorElement x /= TensorElement y = variableValue x E./= variableValue y

instance (Context s, Shape sh, DType a, Fractional a, Fractional (T.TensorElement a), E.Ord (T.TensorElement a)) => E.Ord (Tensor s sh a) where
  Tensor x < Tensor y = variableValue x E.< variableValue y
  Tensor x <= Tensor y = variableValue x E.<= variableValue y
  Tensor x > Tensor y = variableValue x E.> variableValue y
  Tensor x >= Tensor y = variableValue x E.>= variableValue y
  min (Tensor x) (Tensor y) = Tensor (extremumVariable E.min (E.>) x y)
  max (Tensor x) (Tensor y) = Tensor (extremumVariable E.max (E.<) x y)

instance (Context s, DType a, Fractional (T.TensorElement a), E.Ord (T.TensorElement a)) => E.Ord (TensorElement s a) where
  TensorElement x < TensorElement y = variableValue x E.< variableValue y
  TensorElement x <= TensorElement y = variableValue x E.<= variableValue y
  TensorElement x > TensorElement y = variableValue x E.> variableValue y
  TensorElement x >= TensorElement y = variableValue x E.>= variableValue y
  min (TensorElement x) (TensorElement y) = TensorElement (extremumVariable E.min (E.>) x y)
  max (TensorElement x) (TensorElement y) = TensorElement (extremumVariable E.max (E.<) x y)

instance (E.RealFloat (T.Tensor sh a)) => E.RealFloat (Tensor s sh a) where
  isNaN (Tensor x) = E.isNaN (variableValue x)
  isInfinite (Tensor x) = E.isInfinite (variableValue x)

instance (E.RealFloat (T.TensorElement a)) => E.RealFloat (TensorElement s a) where
  isNaN (TensorElement x) = E.isNaN (variableValue x)
  isInfinite (TensorElement x) = E.isInfinite (variableValue x)

instance (Context s, DType a, Num (T.TensorElement a)) => Scalar (Tensor s) a where
  scalarOp (TensorElement x) = Tensor (liftUnary add (\value -> (C.scalar value, I.index @'[])) x)
