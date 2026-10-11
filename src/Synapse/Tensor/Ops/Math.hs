{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Tensor math operations.
module Synapse.Tensor.Ops.Math (dot, matmul) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Construction (full)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Elementwise (map, zipWith)
import qualified Synapse.Tensor.Ops.Reduction as R
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shape, shapeSize)
import Synapse.Tensor.Shapes.Matmul (MatmulShape)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))
import Prelude hiding (map, zip, zipWith)

-- | Inner product of equally sized vectors, returning zero for empty vectors.
dot :: (Shape '[n], DType a, Num (TensorElement a)) => Tensor '[n] a -> Tensor '[n] a -> TensorElement a
dot lhs rhs = R.sum (zipWith (*) lhs rhs)

-- | Store each column contiguously so an inner product reads both inputs sequentially.
-- The compute boundary materializes only the rearranged right-hand input.
packColumns :: (DType a) => Int -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
packColumns shared cols input
  | cols == 1 = input
  | otherwise = A.compute $ A.backpermute (A.shape input) sourceIndex input
  where
    sourceIndex ix =
      let linear = A.unindex1 ix
          columnIndex = linear `quot` A.constant shared
          inner = linear `rem` A.constant shared
          batch = columnIndex `quot` A.constant cols
          col = columnIndex `rem` A.constant cols
       in A.index1 ((batch * A.constant shared + inner) * A.constant cols + col)

-- | Reduce generated products using the backend's array reduction implementation.
-- Products have one consumer and no explicit materialisation boundary.
matrixProduct :: forall a. (DType a, Num (TensorElement a)) => Int -> Int -> Int -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
matrixProduct count rows shared cols left right = A.fold add zero products
  where
    zero = unTensorElement (0 :: TensorElement a)
    add :: A.Exp a -> A.Exp a -> A.Exp a
    add x y = unTensorElement (TensorElement x + TensorElement y)
    packedRight = packColumns shared cols right
    products = A.generate (A.constant (A.Z A.:. count A.:. shared)) $ \ix ->
      let linear = A.indexHead (A.indexTail ix)
          inner = A.indexHead ix
          columnIndex = linear `quot` A.constant cols
          row = columnIndex `rem` A.constant rows
          batch = columnIndex `quot` A.constant rows
          col = linear `rem` A.constant cols
          leftOffset = (batch * A.constant rows + row) * A.constant shared
          rightOffset = (batch * A.constant cols + col) * A.constant shared
          x = TensorElement (left A.! A.index1 (leftOffset + inner))
          y = TensorElement (packedRight A.! A.index1 (rightOffset + inner))
       in unTensorElement (x * y)

-- | Multiply the last two dimensions of tensors with identical leading batch shapes.
-- Broadcasting is explicit and both inputs must have rank at least two.
-- A zero inner dimension produces zeros and an empty output shape produces an empty tensor.
-- Columns of the right input are packed contiguously and products are consumed by an array reduction.
-- Floating point accumulation order is determined by the backend reduction.
matmul ::
  forall lhs rhs a.
  (Shape lhs, Shape rhs, Shape (MatmulShape lhs rhs), DType a, Num (TensorElement a)) =>
  Tensor lhs a ->
  Tensor rhs a ->
  Tensor (MatmulShape lhs rhs) a
matmul (Tensor left) (Tensor right)
  | shapeSize @(MatmulShape lhs rhs) == 0 = Tensor (A.use (A.fromList outputShape []))
  | otherwise = case (reverse (shape @lhs), reverse (shape @rhs)) of
      (0 : _ : _, _ : _ : _) ->
        Tensor (A.fill (A.constant outputShape) (unTensorElement (0 :: TensorElement a)))
      (shared : rows : _, cols : _ : _) ->
        Tensor $ A.reshape (A.constant outputShape) $
          matrixProduct (shapeSize @(MatmulShape lhs rhs)) rows shared cols leftVector rightVector
      _ -> error "Synapse.Tensor.matmul: inconsistent input rank"
  where
    outputShape = backendShape (Proxy @(MatmulShape lhs rhs))
    leftVector = A.flatten left
    rightVector = A.flatten right

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
