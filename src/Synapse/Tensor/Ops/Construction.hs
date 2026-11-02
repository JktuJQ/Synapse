{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Tensor construction and initialisation.
module Synapse.Tensor.Ops.Construction
  ( arange,
    eye,
    fill,
    fromGen,
    full,
    fullLike,
    identity,
    linspace,
    logspace,
    ones,
    onesLike,
    scalar,
    zeros,
    zerosLike,
  )
where

import Synapse.Tensor.Ops (TensorOps (fromElement, literal))
import qualified Synapse.Tensor.Ops.Primitives.Construction as Raw
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Element, FromIntegral, TensorElement, TensorIndex)

-- | Turn an element expression into a rank-zero tensor, preserving its differentiation history.
-- The element expression must use the representation associated with the result tensor.
-- To create a scalar tensor from a host value, use @full \@'[] value@.
scalar ::
  forall {t} a.
  (TensorOps t a) =>
  Element t a ->
  t '[] a
scalar = fromElement

-- | Generate each element from its complete multidimensional index.
-- The callback builds element expressions.
-- Use @coordinate \@axis@ to read an axis coordinate.
-- Generated tensors are constants with respect to differentiation.
-- Scalar indices have no coordinates; empty tensors have no elements.
--
-- @
-- fromGen \@'[2, 3] (\ix -> coordinate \@0 ix * 10 + coordinate \@1 ix)
-- @
fromGen ::
  forall {t} sh a.
  (TensorOps t a, Shape sh) =>
  (TensorIndex sh -> TensorElement a) ->
  t sh a
fromGen = literal . Raw.fromGen

-- | Create a tensor filled with the same value in every element.
full ::
  forall {t} sh a.
  (TensorOps t a, Shape sh) =>
  a ->
  t sh a
full = literal . Raw.full

-- | Alias for 'full'.
fill ::
  forall {t} sh a.
  (TensorOps t a, Shape sh) =>
  a ->
  t sh a
fill = full @sh

-- | Create a tensor filled with zeros.
zeros ::
  forall {t} sh a.
  (TensorOps t a, Shape sh, Num a) =>
  t sh a
zeros = full @sh 0

-- | Create a tensor filled with ones.
ones ::
  forall {t} sh a.
  (TensorOps t a, Shape sh, Num a) =>
  t sh a
ones = full @sh 1

-- | Create a tensor with the same shape as another tensor, filled with a value.
fullLike ::
  forall {t} sh a b.
  (TensorOps t a, Shape sh) =>
  t sh b ->
  a ->
  t sh a
fullLike _ value = literal (Raw.full @sh value)

-- | Create a zero tensor with the same shape as another tensor.
zerosLike ::
  forall {t} sh a b.
  (TensorOps t a, Shape sh, Num a) =>
  t sh b ->
  t sh a
zerosLike _ = zeros @sh

-- | Create a ones tensor with the same shape as another tensor.
onesLike ::
  forall {t} sh a b.
  (TensorOps t a, Shape sh, Num a) =>
  t sh b ->
  t sh a
onesLike _ = ones @sh

-- | Create a vector with evenly spaced values @start, start + step, ...@.
arange ::
  forall {t} n a.
  (TensorOps t a, Shape '[n], Num (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) =>
  a ->
  a ->
  t '[n] a
arange x y = literal $ Raw.arange x y

-- | Create a vector with @n@ evenly spaced values from @start@ to @end@.
-- For @n >= 2@ both endpoints are taken directly from the supplied values.
-- Interior points use weighted endpoints to avoid overflowing their difference.
-- Zero elements produce an empty vector; one element produces @start@.
-- Descending endpoints are supported.
linspace ::
  forall {t} n a.
  (TensorOps t a, Shape '[n], Fractional a, Fractional (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) =>
  a ->
  a ->
  t '[n] a
linspace x y = literal $ Raw.linspace x y

-- | Exponentiate 'linspace' using the supplied base.
-- Parameters are start exponent, end exponent, then base.
-- Empty and singleton cases follow 'linspace'.
logspace ::
  forall {t} n a.
  (TensorOps t a, Shape '[n], Floating a, Floating (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) =>
  a ->
  a ->
  a ->
  t '[n] a
logspace x y z = literal $ Raw.logspace x y z

-- | Create a matrix with ones on the main diagonal and zeros elsewhere.
eye ::
  forall {t} rows cols a.
  (TensorOps t a, Shape '[rows, cols], Num (TensorElement a)) =>
  t '[rows, cols] a
eye = literal Raw.eye

-- | Create a square identity matrix.
identity ::
  forall {t} n a.
  (TensorOps t a, Shape '[n, n], Num (TensorElement a)) =>
  t '[n, n] a
identity = eye @n @n
