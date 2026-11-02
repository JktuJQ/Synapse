{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Inner products and batched matrix multiplication.
module Synapse.Tensor.Ops.Math (dot, matmul) where

import qualified Synapse.Autograd.Tensor.Ops.Primitives.Math as Pullback
import Synapse.Tensor.Ops (TensorOps (..))
import qualified Synapse.Tensor.Ops.Primitives.Math as Raw
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Matmul (MatmulShape)
import Synapse.Tensor.Type (TensorElement)

-- | Inner product of equally sized vectors, returning zero for empty vectors.
dot ::
  forall {t} n a.
  (TensorOps t a, Num (TensorElement a), Shape '[n]) =>
  t '[n] a ->
  t '[n] a ->
  t '[] a
dot = lift2 Raw.dot Pullback.dot

-- | Multiply vectors and matrices, broadcasting leading batch dimensions.
-- Vector operands omit their promoted matrix axis from the result; two vectors produce a scalar.
-- Scalar operands are rejected.
-- A zero inner dimension produces zeros and an empty output shape produces an empty tensor.
-- Batch axes align from the right and corresponding dimensions must be equal or singleton.
-- Floating point accumulation order is determined by the backend reduction.
matmul ::
  forall {t} lhs rhs a.
  (TensorOps t a, Num (TensorElement a), Shape lhs, Shape rhs, Shape (MatmulShape lhs rhs)) =>
  t lhs a ->
  t rhs a ->
  t (MatmulShape lhs rhs) a
matmul = lift2 Raw.matmul Pullback.matmul
