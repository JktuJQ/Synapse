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
import Synapse.Tensor.Ops (Needs (Fractional), TensorOps (..))
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
  (+) =
    curry $
      customVJP
        Fractional
        (uncurry (+))
        ( \(left, right) ->
            let output = (+) left right
             in (output, ((left, right), output))
        )
        (\((left, right), output) -> MathPullback.add left right output)
  (-) =
    curry $
      customVJP
        Fractional
        (uncurry (-))
        ( \(left, right) ->
            let output = (-) left right
             in (output, ((left, right), output))
        )
        ( \((left, right), output) ->
            MathPullback.subtract left right output
        )
  (*) =
    curry $
      customVJP
        Fractional
        (uncurry (*))
        ( \(left, right) ->
            let output = (*) left right
             in (output, ((left, right), output))
        )
        ( \((left, right), output) ->
            MathPullback.multiply left right output
        )
  negate = customVJP Fractional negate (\primalInput -> let output = negate primalInput in (output, (primalInput, output))) (uncurry MathPullback.negate)
  abs = customVJP Fractional abs (\primalInput -> let output = abs primalInput in (output, (primalInput, output))) (uncurry MathPullback.abs)
  signum = customVJP Fractional signum (\primalInput -> let output = signum primalInput in (output, (primalInput, output))) (uncurry (MathPullback.signum (const 0)))
  fromInteger = literal . fromInteger

instance (Context s, Shape sh, DType a, Fractional a, Fractional (T.TensorElement a)) => Fractional (Tensor s sh a) where
  (/) =
    curry $
      customVJP
        Fractional
        (uncurry (/))
        ( \(left, right) ->
            let output = (/) left right
             in (output, ((left, right), output))
        )
        ( \((left, right), output) ->
            MathPullback.divide left right output
        )
  recip = customVJP Fractional recip (\primalInput -> let output = recip primalInput in (output, (primalInput, output))) (uncurry MathPullback.recip)
  fromRational = literal . fromRational

instance (Context s, Shape sh, DType a, Floating a, Floating (T.TensorElement a)) => Floating (Tensor s sh a) where
  pi = literal pi
  exp = customVJP Fractional exp (\primalInput -> let output = exp primalInput in (output, (primalInput, output))) (uncurry MathPullback.exp)
  log = customVJP Fractional log (\primalInput -> let output = log primalInput in (output, (primalInput, output))) (uncurry MathPullback.log)
  sqrt = customVJP Fractional sqrt (\primalInput -> let output = sqrt primalInput in (output, (primalInput, output))) (uncurry MathPullback.sqrt)
  sin = customVJP Fractional sin (\primalInput -> let output = sin primalInput in (output, (primalInput, output))) (uncurry MathPullback.sin)
  cos = customVJP Fractional cos (\primalInput -> let output = cos primalInput in (output, (primalInput, output))) (uncurry MathPullback.cos)
  tan = customVJP Fractional tan (\primalInput -> let output = tan primalInput in (output, (primalInput, output))) (uncurry MathPullback.tan)
  asin = customVJP Fractional asin (\primalInput -> let output = asin primalInput in (output, (primalInput, output))) (uncurry MathPullback.asin)
  acos = customVJP Fractional acos (\primalInput -> let output = acos primalInput in (output, (primalInput, output))) (uncurry MathPullback.acos)
  atan = customVJP Fractional atan (\primalInput -> let output = atan primalInput in (output, (primalInput, output))) (uncurry MathPullback.atan)
  sinh = customVJP Fractional sinh (\primalInput -> let output = sinh primalInput in (output, (primalInput, output))) (uncurry MathPullback.sinh)
  cosh = customVJP Fractional cosh (\primalInput -> let output = cosh primalInput in (output, (primalInput, output))) (uncurry MathPullback.cosh)
  tanh = customVJP Fractional tanh (\primalInput -> let output = tanh primalInput in (output, (primalInput, output))) (uncurry MathPullback.tanh)
  asinh = customVJP Fractional asinh (\primalInput -> let output = asinh primalInput in (output, (primalInput, output))) (uncurry MathPullback.asinh)
  acosh = customVJP Fractional acosh (\primalInput -> let output = acosh primalInput in (output, (primalInput, output))) (uncurry MathPullback.acosh)
  atanh = customVJP Fractional atanh (\primalInput -> let output = atanh primalInput in (output, (primalInput, output))) (uncurry MathPullback.atanh)
  (**) =
    curry $
      customVJP
        Fractional
        (uncurry (**))
        ( \(left, right) ->
            let output = (**) left right
             in (output, ((left, right), output))
        )
        (\((left, right), output) -> MathPullback.power left right output)
  logBase x y = log y / log x

deriving instance (Context s, Num (T.TensorElement a)) => Num (TensorElement s a)

deriving instance (Context s, Fractional (T.TensorElement a)) => Fractional (TensorElement s a)

deriving instance (Context s, Floating (T.TensorElement a)) => Floating (TensorElement s a)

instance (Context s, Shape sh, DType a, Fractional a, Fractional (T.TensorElement a)) => E.Conditional (Tensor s sh a) where
  type Condition (Tensor s sh a) = T.Tensor sh Bool
  type Predicate (Tensor s sh a) = T.Tensor '[] Bool
  select mask =
    curry $
      customVJP
        Fractional
        (uncurry (E.select mask))
        ( \(left, right) ->
            let output = E.select mask left right
             in (output, ((left, right), output))
        )
        ( \((left, right), output) ->
            Pullback.select mask left right output
        )
  cond flag =
    curry $
      customVJP
        Fractional
        (uncurry (E.cond flag))
        ( \(left, right) ->
            let output = E.cond flag left right
             in (output, ((left, right), output))
        )
        ( \((left, right), output) ->
            Pullback.cond flag left right output
        )

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
  min =
    curry $
      customVJP
        Fractional
        (uncurry E.min)
        ( \(left, right) ->
            let output = E.min left right
             in (output, ((left, right), output))
        )
        ( \((left, right), output) ->
            Pullback.extremum (E.>) left right output
        )
  max =
    curry $
      customVJP
        Fractional
        (uncurry E.max)
        ( \(left, right) ->
            let output = E.max left right
             in (output, ((left, right), output))
        )
        ( \((left, right), output) ->
            Pullback.extremum (E.<) left right output
        )

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
