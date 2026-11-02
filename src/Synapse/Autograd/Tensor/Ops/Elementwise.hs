{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Arithmetic, comparisons, and conditional selection for differentiable values.
module Synapse.Autograd.Tensor.Ops.Elementwise () where

import qualified Numeric.Backprop.Explicit as BPE
import Synapse.Autograd.Tensor.Ops ()
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Elementwise as Pullback
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Math as MathPullback
import Synapse.Autograd.Tensor.Type (Tensor (..), TensorElement (..))
import Synapse.Autograd.Variable (Backpropagatable, Context, Var (..), add, liftBinary, liftUnary, variableValue, zero)
import Synapse.Tensor.Ops (TensorOps (..))
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (DType)
import qualified Synapse.Tensor.Type as T

-- | Register a scalar primitive with its cached forward result and input accumulation rule.
unaryVariable :: (Context s, Backpropagatable a) => (a -> a) -> (a -> a -> a -> a) -> Var s a -> Var s a
unaryVariable forward backward = liftUnary add $ \x ->
  let output = forward x
   in (output, backward x output)

-- | Register a binary scalar primitive with one contribution per input.
binaryVariable :: (Context s, Backpropagatable a) => (a -> a -> a) -> (a -> a -> a -> a -> (a, a)) -> Var s a -> Var s a -> Var s a
binaryVariable forward backward = liftBinary add add $ \x y ->
  let output = forward x y
   in (output, backward x y output)

instance (Context s, Backpropagatable a, Num a) => Num (Var s a) where
  (+) = binaryVariable (+) MathPullback.add
  (-) = binaryVariable (-) MathPullback.subtract
  (*) = binaryVariable (*) MathPullback.multiply
  negate = unaryVariable negate MathPullback.negate
  abs = unaryVariable abs MathPullback.abs
  signum = unaryVariable signum (MathPullback.signum zero)
  fromInteger = Var . BPE.auto . fromInteger

instance (Context s, Backpropagatable a, Fractional a) => Fractional (Var s a) where
  (/) = binaryVariable (/) MathPullback.divide
  recip = unaryVariable recip MathPullback.recip
  fromRational = Var . BPE.auto . fromRational

instance (Context s, Backpropagatable a, Floating a) => Floating (Var s a) where
  pi = Var (BPE.auto pi)
  exp = unaryVariable exp MathPullback.exp
  log = unaryVariable log MathPullback.log
  sqrt = unaryVariable sqrt MathPullback.sqrt
  sin = unaryVariable sin MathPullback.sin
  cos = unaryVariable cos MathPullback.cos
  tan = unaryVariable tan MathPullback.tan
  asin = unaryVariable asin MathPullback.asin
  acos = unaryVariable acos MathPullback.acos
  atan = unaryVariable atan MathPullback.atan
  sinh = unaryVariable sinh MathPullback.sinh
  cosh = unaryVariable cosh MathPullback.cosh
  tanh = unaryVariable tanh MathPullback.tanh
  asinh = unaryVariable asinh MathPullback.asinh
  acosh = unaryVariable acosh MathPullback.acosh
  atanh = unaryVariable atanh MathPullback.atanh
  (**) = binaryVariable (**) MathPullback.power
  logBase x y = log y / log x

instance (Context s, Shape sh, DType a, Fractional a, Fractional (T.TensorElement a)) => Num (Tensor s sh a) where
  (+) = lift2 (+) MathPullback.add
  (-) = lift2 (-) MathPullback.subtract
  (*) = lift2 (*) MathPullback.multiply
  negate = lift1 negate MathPullback.negate
  abs = lift1 abs MathPullback.abs
  signum = lift1 signum (MathPullback.signum (const 0))
  fromInteger = literal . fromInteger

instance (Context s, Shape sh, DType a, Fractional a, Fractional (T.TensorElement a)) => Fractional (Tensor s sh a) where
  (/) = lift2 (/) MathPullback.divide
  recip = lift1 recip MathPullback.recip
  fromRational = literal . fromRational

instance (Context s, Shape sh, DType a, Floating a, Floating (T.TensorElement a)) => Floating (Tensor s sh a) where
  pi = literal pi
  exp = lift1 exp MathPullback.exp
  log = lift1 log MathPullback.log
  sqrt = lift1 sqrt MathPullback.sqrt
  sin = lift1 sin MathPullback.sin
  cos = lift1 cos MathPullback.cos
  tan = lift1 tan MathPullback.tan
  asin = lift1 asin MathPullback.asin
  acos = lift1 acos MathPullback.acos
  atan = lift1 atan MathPullback.atan
  sinh = lift1 sinh MathPullback.sinh
  cosh = lift1 cosh MathPullback.cosh
  tanh = lift1 tanh MathPullback.tanh
  asinh = lift1 asinh MathPullback.asinh
  acosh = lift1 acosh MathPullback.acosh
  atanh = lift1 atanh MathPullback.atanh
  (**) = lift2 (**) MathPullback.power
  logBase x y = log y / log x

deriving instance (Context s, Num (T.TensorElement a)) => Num (TensorElement s a)

deriving instance (Context s, Fractional (T.TensorElement a)) => Fractional (TensorElement s a)

deriving instance (Context s, Floating (T.TensorElement a)) => Floating (TensorElement s a)

instance (Context s, Shape sh, DType a, Fractional a, Fractional (T.TensorElement a)) => E.Conditional (Tensor s sh a) where
  type Condition (Tensor s sh a) = T.Tensor sh Bool
  type Predicate (Tensor s sh a) = T.Tensor '[] Bool
  select mask = lift2 (E.select mask) (Pullback.select mask)
  cond flag = lift2 (E.cond flag) (Pullback.cond flag)

instance (Context s, DType a, Num (T.TensorElement a)) => E.Conditional (TensorElement s a) where
  type Condition (TensorElement s a) = T.TensorElement Bool
  type Predicate (TensorElement s a) = T.TensorElement Bool
  select mask (TensorElement x) (TensorElement y) = TensorElement (binaryVariable (E.select mask) (Pullback.select mask) x y)
  cond flag (TensorElement x) (TensorElement y) = TensorElement (binaryVariable (E.cond flag) (Pullback.cond flag) x y)

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
  min = lift2 E.min (Pullback.extremum (E.>))
  max = lift2 E.max (Pullback.extremum (E.<))

instance (Context s, DType a, Fractional (T.TensorElement a), E.Ord (T.TensorElement a)) => E.Ord (TensorElement s a) where
  TensorElement x < TensorElement y = variableValue x E.< variableValue y
  TensorElement x <= TensorElement y = variableValue x E.<= variableValue y
  TensorElement x > TensorElement y = variableValue x E.> variableValue y
  TensorElement x >= TensorElement y = variableValue x E.>= variableValue y
  min (TensorElement x) (TensorElement y) = TensorElement (binaryVariable E.min (Pullback.extremum (E.>)) x y)
  max (TensorElement x) (TensorElement y) = TensorElement (binaryVariable E.max (Pullback.extremum (E.<)) x y)

instance (E.RealFloat (T.Tensor sh a)) => E.RealFloat (Tensor s sh a) where
  isNaN (Tensor x) = E.isNaN (variableValue x)
  isInfinite (Tensor x) = E.isInfinite (variableValue x)

instance (E.RealFloat (T.TensorElement a)) => E.RealFloat (TensorElement s a) where
  isNaN (TensorElement x) = E.isNaN (variableValue x)
  isInfinite (TensorElement x) = E.isInfinite (variableValue x)
