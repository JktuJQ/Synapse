{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Tensor construction and initialisation.
module Synapse.Tensor.Construction
  ( Scalar (..),
    Construction (..),
    arange,
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

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Elementwise.Helpers (map)
import Synapse.Tensor.Ops.Index (coordinate)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shapeSize)
import Synapse.Tensor.Type (Element, FromIntegral, Tensor (..), TensorElement (..), TensorIndex (..), constant)
import qualified Synapse.Tensor.Type as E
import Prelude hiding (map)

-- | Embedding an element expression in a scalar tensor.
class (DType a) => Scalar t a where
  -- | Preserve an element expression as a rank-zero tensor.
  scalarOp :: Element t a -> t '[] a

instance (DType a) => Scalar Tensor a where
  scalarOp (TensorElement x) = Tensor (A.unit x)

-- | Construction of tensor expressions from values and element generators.
class Construction t where
  -- | Generate each element from its complete multidimensional index.
  -- The callback builds element expressions.
  -- Use @coordinate \@axis@ to read an axis coordinate.
  -- Scalar indices have no coordinates; empty tensors have no elements.
  --
  -- @
  -- fromGen \@'[2, 3] (\ix -> coordinate \@0 ix * 10 + coordinate \@1 ix)
  -- @
  fromGenOp ::
    forall sh a.
    (Shape sh, DType a) =>
    (TensorIndex sh -> Element t a) ->
    t sh a

  -- | Create a tensor filled with the same value in every element.
  fullOp ::
    forall sh a.
    (Shape sh, DType a) =>
    a ->
    t sh a

  -- | Alias for 'full'.
  fillOp ::
    forall sh a.
    (Shape sh, DType a) =>
    a ->
    t sh a

  -- | Create a tensor filled with zeros.
  zerosOp ::
    forall sh a.
    (Shape sh, DType a, Num a) =>
    t sh a

  -- | Create a tensor filled with ones.
  onesOp ::
    forall sh a.
    (Shape sh, DType a, Num a) =>
    t sh a

  -- | Create a tensor with the same shape as another tensor, filled with a value.
  fullLikeOp ::
    forall sh a b.
    (Shape sh, DType a) =>
    t sh b ->
    a ->
    t sh a

  -- | Create a zero tensor with the same shape as another tensor.
  zerosLikeOp ::
    forall sh a b.
    (Shape sh, DType a, Num a) =>
    t sh b ->
    t sh a

  -- | Create a ones tensor with the same shape as another tensor.
  onesLikeOp ::
    forall sh a b.
    (Shape sh, DType a, Num a) =>
    t sh b ->
    t sh a

  -- | Create a vector with evenly spaced values @start, start + step, ...@.
  arangeOp ::
    forall n a.
    (Shape '[n], DType a, Num (Element t a), FromIntegral (Element t Int) (Element t a)) =>
    a ->
    a ->
    t '[n] a

  -- | Create a vector with @n@ evenly spaced values from @start@ to @end@.
  -- For @n >= 2@ both endpoints are taken directly from the supplied values.
  -- Interior points use weighted endpoints to avoid overflowing their difference.
  -- Zero elements produce an empty vector; one element produces @start@.
  -- Descending endpoints are supported.
  linspaceOp ::
    forall n a.
    (Shape '[n], DType a, Fractional a, Fractional (Element t a), FromIntegral (Element t Int) (Element t a)) =>
    a ->
    a ->
    t '[n] a

  -- | Exponentiate 'linspace' using the supplied base.
  -- Parameters are start exponent, end exponent, then base.
  -- Empty and singleton cases follow 'linspace'.
  logspaceOp ::
    forall n a.
    (Shape '[n], DType a, Floating a, Floating (Element t a), FromIntegral (Element t Int) (Element t a)) =>
    a ->
    a ->
    a ->
    t '[n] a

  -- | Create a matrix with ones on the main diagonal and zeros elsewhere.
  eyeOp ::
    forall rows cols a.
    (Shape '[rows, cols], DType a, Num (Element t a)) =>
    t '[rows, cols] a

  -- | Create a square identity matrix.
  identityOp ::
    forall n a.
    (Shape '[n, n], DType a, Num (Element t a)) =>
    t '[n, n] a

instance Construction Tensor where
  fromGenOp ::
    forall sh a.
    (Shape sh, DType a) =>
    (TensorIndex sh -> TensorElement a) ->
    Tensor sh a
  fromGenOp f = Tensor $ A.generate dims $ \ix ->
    unTensorElement (f (TensorIndex ix))
    where
      dims = A.constant (backendShape (Proxy @sh))

  fullOp ::
    forall sh a.
    (Shape sh, DType a) =>
    a ->
    Tensor sh a
  fullOp x = Tensor $ A.fill (A.constant (backendShape (Proxy @sh))) (A.constant x)

  fillOp ::
    forall sh a.
    (Shape sh, DType a) =>
    a ->
    Tensor sh a
  fillOp = full @sh

  zerosOp ::
    forall sh a.
    (Shape sh, DType a, Num a) =>
    Tensor sh a
  zerosOp = full @sh 0

  onesOp ::
    forall sh a.
    (Shape sh, DType a, Num a) =>
    Tensor sh a
  onesOp = full @sh 1

  fullLikeOp ::
    forall sh a b.
    (Shape sh, DType a) =>
    Tensor sh b ->
    a ->
    Tensor sh a
  fullLikeOp _ = full @sh

  zerosLikeOp ::
    forall sh a b.
    (Shape sh, DType a, Num a) =>
    Tensor sh b ->
    Tensor sh a
  zerosLikeOp _ = zeros @sh

  onesLikeOp ::
    forall sh a b.
    (Shape sh, DType a, Num a) =>
    Tensor sh b ->
    Tensor sh a
  onesLikeOp _ = ones @sh

  arangeOp ::
    forall n a.
    (Shape '[n], DType a, Num (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) =>
    a ->
    a ->
    Tensor '[n] a
  arangeOp start step =
    fromGen @'[n] $ \ix -> constant start + E.fromIntegral (coordinate @0 ix :: TensorElement Int) * constant step

  linspaceOp ::
    forall n a.
    (Shape '[n], DType a, Fractional a, Fractional (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) =>
    a ->
    a ->
    Tensor '[n] a
  linspaceOp start end =
    case shapeSize @'[n] of
      0 -> full @'[n] start
      1 -> full @'[n] start
      count ->
        fromGen @'[n] $ \ix ->
          let i = unTensorElement (coordinate @0 ix)
              weight :: TensorElement a
              weight = E.fromIntegral (coordinate @0 ix :: TensorElement Int) / constant (fromIntegral (count - 1))
              value = (1 - weight) * constant start + weight * constant end
           in TensorElement $
                A.cond (i A.== 0) (A.constant start) $
                  A.cond (i A.== A.constant (count - 1)) (A.constant end) (unTensorElement value)

  logspaceOp ::
    forall n a.
    (Shape '[n], DType a, Floating a, Floating (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) =>
    a ->
    a ->
    a ->
    Tensor '[n] a
  logspaceOp start end base =
    map (constant base **) (linspace @n @a start end)

  eyeOp ::
    forall rows cols a.
    (Shape '[rows, cols], DType a, Num (TensorElement a)) =>
    Tensor '[rows, cols] a
  eyeOp =
    fromGen @'[rows, cols] $ \ix ->
      let row = unTensorElement (coordinate @0 ix)
          col = unTensorElement (coordinate @1 ix)
       in TensorElement (A.cond (row A.== col) (unTensorElement (1 :: TensorElement a)) (unTensorElement (0 :: TensorElement a)))

  identityOp ::
    forall n a.
    (Shape '[n, n], DType a, Num (TensorElement a)) =>
    Tensor '[n, n] a
  identityOp = eye @n @n

-- | Turn an element expression into a rank-zero tensor for execution.
-- Numeric literals work directly; use 'constant' for existing host values.
-- Differentiable element expressions retain their gradient history.
scalar ::
  forall {t} a.
  (Scalar t a) =>
  Element t a ->
  t '[] a
scalar = scalarOp @t @a

-- | Generate each element from its complete multidimensional index.
-- The callback builds element expressions.
-- Use @coordinate \@axis@ to read an axis coordinate.
-- Scalar indices have no coordinates; empty tensors have no elements.
--
-- @
-- fromGen \@'[2, 3] (\ix -> coordinate \@0 ix * 10 + coordinate \@1 ix)
-- @
fromGen ::
  forall {t} sh a.
  (Construction t, Shape sh, DType a) =>
  (TensorIndex sh -> Element t a) ->
  t sh a
fromGen = fromGenOp @t @sh @a

-- | Create a tensor filled with the same value in every element.
full ::
  forall {t} sh a.
  (Construction t, Shape sh, DType a) =>
  a ->
  t sh a
full = fullOp @t @sh @a

-- | Alias for 'full'.
fill ::
  forall {t} sh a.
  (Construction t, Shape sh, DType a) =>
  a ->
  t sh a
fill = fillOp @t @sh @a

-- | Create a tensor filled with zeros.
zeros ::
  forall {t} sh a.
  (Construction t, Shape sh, DType a, Num a) =>
  t sh a
zeros = zerosOp @t @sh @a

-- | Create a tensor filled with ones.
ones ::
  forall {t} sh a.
  (Construction t, Shape sh, DType a, Num a) =>
  t sh a
ones = onesOp @t @sh @a

-- | Create a tensor with the same shape as another tensor, filled with a value.
fullLike ::
  forall {t} sh a b.
  (Construction t, Shape sh, DType a) =>
  t sh b ->
  a ->
  t sh a
fullLike = fullLikeOp @t @sh @a @b

-- | Create a zero tensor with the same shape as another tensor.
zerosLike ::
  forall {t} sh a b.
  (Construction t, Shape sh, DType a, Num a) =>
  t sh b ->
  t sh a
zerosLike = zerosLikeOp @t @sh @a @b

-- | Create a ones tensor with the same shape as another tensor.
onesLike ::
  forall {t} sh a b.
  (Construction t, Shape sh, DType a, Num a) =>
  t sh b ->
  t sh a
onesLike = onesLikeOp @t @sh @a @b

-- | Create a vector with evenly spaced values @start, start + step, ...@.
arange ::
  forall {t} n a.
  (Construction t, Shape '[n], DType a, Num (Element t a), FromIntegral (Element t Int) (Element t a)) =>
  a ->
  a ->
  t '[n] a
arange = arangeOp @t @n @a

-- | Create a vector with @n@ evenly spaced values from @start@ to @end@.
-- For @n >= 2@ both endpoints are taken directly from the supplied values.
-- Interior points use weighted endpoints to avoid overflowing their difference.
-- Zero elements produce an empty vector; one element produces @start@.
-- Descending endpoints are supported.
linspace ::
  forall {t} n a.
  (Construction t, Shape '[n], DType a, Fractional a, Fractional (Element t a), FromIntegral (Element t Int) (Element t a)) =>
  a ->
  a ->
  t '[n] a
linspace = linspaceOp @t @n @a

-- | Exponentiate 'linspace' using the supplied base.
-- Parameters are start exponent, end exponent, then base.
-- Empty and singleton cases follow 'linspace'.
logspace ::
  forall {t} n a.
  (Construction t, Shape '[n], DType a, Floating a, Floating (Element t a), FromIntegral (Element t Int) (Element t a)) =>
  a ->
  a ->
  a ->
  t '[n] a
logspace = logspaceOp @t @n @a

-- | Create a matrix with ones on the main diagonal and zeros elsewhere.
eye ::
  forall {t} rows cols a.
  (Construction t, Shape '[rows, cols], DType a, Num (Element t a)) =>
  t '[rows, cols] a
eye = eyeOp @t @rows @cols @a

-- | Create a square identity matrix.
identity ::
  forall {t} n a.
  (Construction t, Shape '[n, n], DType a, Num (Element t a)) =>
  t '[n, n] a
identity = identityOp @t @n @a
