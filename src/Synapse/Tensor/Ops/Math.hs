{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Tensor math operations.
module Synapse.Tensor.Ops.Math (LinearAlgebra (..), dot, matmul) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Construction (full)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Elementwise.Helpers (map, zipWith)
import Synapse.Tensor.Ops.Math.Helpers (broadcastDimensions, matrixDimensions, matrixProduct)
import qualified Synapse.Tensor.Ops.Reduction as R
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shape)
import Synapse.Tensor.Shapes.Matmul (MatmulShape)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))
import Prelude hiding (map, zip, zipWith)

-- | Vector and matrix products with statically checked shapes.
class (DType a) => LinearAlgebra t a where
  -- | Inner product of equally sized vectors, returning zero for empty vectors.
  dotOp ::
    forall n.
    (Shape '[n]) =>
    t '[n] a ->
    t '[n] a ->
    t '[] a

  -- | Multiply vectors and matrices, broadcasting leading batch dimensions.
  -- Vector operands omit their promoted matrix axis from the result; two vectors produce a scalar.
  -- Scalar operands are rejected.
  -- A zero inner dimension produces zeros and an empty output shape produces an empty tensor.
  -- Batch axes align from the right and corresponding dimensions must be equal or singleton.
  -- Floating point accumulation order is determined by the backend reduction.
  matmulOp ::
    forall lhs rhs.
    (Shape lhs, Shape rhs, Shape (MatmulShape lhs rhs)) =>
    t lhs a ->
    t rhs a ->
    t (MatmulShape lhs rhs) a

instance (DType a, Num (TensorElement a)) => LinearAlgebra Tensor a where
  dotOp lhs rhs = R.sum (zipWith (*) lhs rhs)

  matmulOp ::
    forall lhs rhs.
    (Shape lhs, Shape rhs, Shape (MatmulShape lhs rhs)) =>
    Tensor lhs a ->
    Tensor rhs a ->
    Tensor (MatmulShape lhs rhs) a
  matmulOp (Tensor left) (Tensor right) =
    Tensor $
      A.reshape (A.constant (backendShape (Proxy @(MatmulShape lhs rhs)))) $
        matrixProduct leftBatch rightBatch batchDims rows shared cols (A.flatten left) (A.flatten right)
    where
      (leftBatch, rows, shared) = matrixDimensions True (shape @lhs)
      (rightBatch, _, cols) = matrixDimensions False (shape @rhs)
      batchDims = broadcastDimensions leftBatch rightBatch

-- | Inner product of equally sized vectors, returning zero for empty vectors.
dot ::
  forall {t} n a.
  (LinearAlgebra t a, Shape '[n]) =>
  t '[n] a ->
  t '[n] a ->
  t '[] a
dot = dotOp @t @a @n

-- | Multiply vectors and matrices, broadcasting leading batch dimensions.
-- Vector operands omit their promoted matrix axis from the result; two vectors produce a scalar.
-- Scalar operands are rejected.
-- A zero inner dimension produces zeros and an empty output shape produces an empty tensor.
-- Batch axes align from the right and corresponding dimensions must be equal or singleton.
-- Floating point accumulation order is determined by the backend reduction.
matmul ::
  forall {t} lhs rhs a.
  (LinearAlgebra t a, Shape lhs, Shape rhs, Shape (MatmulShape lhs rhs)) =>
  t lhs a ->
  t rhs a ->
  t (MatmulShape lhs rhs) a
matmul = matmulOp @t @a @lhs @rhs

instance (Shape sh, DType a, Num a, Num (TensorElement a)) => Num (Tensor sh a) where
  (+) = zipWith (+)
  (-) = zipWith (-)
  (*) = zipWith (*)
  negate = map negate
  abs = map abs
  signum = map signum
  fromInteger = full @sh . fromInteger

instance (Shape sh, DType a, Fractional a, Fractional (TensorElement a)) => Fractional (Tensor sh a) where
  (/) = zipWith (/)
  recip = map recip
  fromRational = full @sh . fromRational

instance (Shape sh, DType a, Floating a, Floating (TensorElement a)) => Floating (Tensor sh a) where
  pi = full @sh pi
  exp = map exp
  log = map log
  sqrt = map sqrt
  (**) = zipWith (**)
  logBase = zipWith logBase
  sin = map sin
  cos = map cos
  tan = map tan
  asin = map asin
  acos = map acos
  atan = map atan
  sinh = map sinh
  cosh = map cosh
  tanh = map tanh
  asinh = map asinh
  acosh = map acosh
  atanh = map atanh
