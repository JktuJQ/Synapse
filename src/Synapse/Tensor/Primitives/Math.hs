{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Staged matrix computations and elementwise numeric instances.
module Synapse.Tensor.Primitives.Math (matrixDimensions, broadcastDimensions, broadcastBatchIndex, broadcastCopies, multiply, contract, dot, matmul) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Primitives.Construction (full)
import Synapse.Tensor.Primitives.Elementwise (map, zipWith)
import qualified Synapse.Tensor.Primitives.Reduction as Reduction
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shape, strides)
import Synapse.Tensor.Shapes.Matmul (MatmulShape)
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..))
import Prelude hiding (map, zipWith)
import qualified Prelude as P

-- | Promote a left vector to a row or a right vector to a column.
matrixDimensions :: Bool -> [Int] -> ([Int], Int, Int)
matrixDimensions left dims = case reverse dims of
  [features] -> if left then ([], 1, features) else ([], features, 1)
  cols : rows : batch -> (reverse batch, rows, cols)
  [] -> error "Synapse.Tensor.matmul: scalar operand"

-- | Combine trailing batch dimensions without materializing broadcast copies.
broadcastDimensions :: [Int] -> [Int] -> [Int]
broadcastDimensions lhs rhs = reverse (combine (reverse lhs) (reverse rhs))
  where
    combine [] ys = ys
    combine xs [] = xs
    combine (x : xs) (y : ys)
      | x == 1 = y : combine xs ys
      | y == 1 || x == y = x : combine xs ys
      | otherwise = error "Synapse.Tensor.matmul: incompatible batch dimensions"

-- | Decode a row-major position into one coordinate per dimension.
coordinates :: [Int] -> A.Exp Int -> [A.Exp Int]
coordinates dims linear = P.zipWith (\dim stride -> linear `A.quot` A.constant stride `A.rem` A.constant dim) dims (strides dims)

-- | Project a common batch index onto an operand's original batch dimensions.
broadcastBatchIndex :: [Int] -> [Int] -> A.Exp Int -> A.Exp Int
broadcastBatchIndex source target linear
  | source == target = linear
  | null source = 0
  | otherwise =
      let padding = length target - length source
          sourceDims = replicate padding 1 ++ source
          sourceStrides = replicate padding 0 ++ strides source
       in sum [coordinate * A.constant stride | (dim, stride, coordinate) <- zip3 sourceDims sourceStrides (coordinates target linear), dim /= 1]

-- | Enumerate the common batch positions contributing to one original batch position.
broadcastCopies :: [Int] -> [Int] -> (Int, A.Exp Int -> A.Exp Int -> A.Exp Int)
broadcastCopies source target
  | padded == target = (1, const)
  | otherwise = (product repeatedDims, inject)
  where
    padded = replicate (length target - length source) 1 ++ source
    repeated = [axis | (axis, (dim, size)) <- zip [0 :: Int ..] (zip padded target), dim == 1 && size /= 1]
    repeatedDims = [size | (axis, size) <- zip [0 ..] target, axis `elem` repeated]
    inject original copy =
      let fixed = coordinates padded original
          varying = coordinates repeatedDims copy
          merge [] _ = []
          merge ((axis, coordinate) : rest) remaining
            | axis `elem` repeated = case remaining of
                value : values -> value : merge rest values
                [] -> error "Synapse.Tensor.matmul: inconsistent broadcast coordinates"
            | otherwise = coordinate : merge rest remaining
       in sum (P.zipWith (*) (merge (zip [0 ..] fixed) varying) (P.map A.constant (strides target)))

-- | Multiply scalar expressions using the element arithmetic instance.
multiply :: (Num (TensorElement a)) => A.Exp a -> A.Exp a -> A.Exp a
multiply x y = unTensorElement (TensorElement x * TensorElement y)

-- | Sum one generated row of terms per output element, including empty contractions.
contract :: forall a. (DType a, Num (TensorElement a)) => Int -> Int -> (A.Exp Int -> A.Exp Int -> A.Exp a) -> A.Acc (A.Vector a)
contract count width term
  | count == 0 = A.use (A.fromList (A.Z A.:. 0) [])
  | width == 0 = A.fill (A.constant (A.Z A.:. count)) zero
  | otherwise = A.fold plus zero $ A.generate (A.constant (A.Z A.:. count A.:. width)) $ \ix -> term (A.indexHead (A.indexTail ix)) (A.indexHead ix)
  where
    zero = unTensorElement (0 :: TensorElement a)
    plus :: A.Exp a -> A.Exp a -> A.Exp a
    plus x y = unTensorElement (TensorElement x + TensorElement y)

-- | Store each original right-hand column contiguously without expanding its batch dimensions.
packColumns :: (DType a) => Int -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
packColumns shared cols input
  | cols <= 1 || shared == 0 = input
  | otherwise = A.compute $ A.backpermute (A.shape input) sourceIndex input
  where
    sourceIndex ix =
      let linear = A.unindex1 ix
          columnIndex = linear `A.quot` A.constant shared
          inner = linear `A.rem` A.constant shared
          batch = columnIndex `A.quot` A.constant cols
          col = columnIndex `A.rem` A.constant cols
       in A.index1 ((batch * A.constant shared + inner) * A.constant cols + col)

-- | Contract matrix products while mapping shared batch positions back to each operand.
matrixProduct :: (DType a, Num (TensorElement a)) => [Int] -> [Int] -> [Int] -> Int -> Int -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
matrixProduct leftBatch rightBatch batchDims rows shared cols left right =
  contract (product batchDims * rows * cols) shared $ \linear inner ->
    let rowIndex = linear `A.quot` A.constant cols
        row = rowIndex `A.rem` A.constant rows
        batch = rowIndex `A.quot` A.constant rows
        col = linear `A.rem` A.constant cols
        leftOffset = (broadcastBatchIndex leftBatch batchDims batch * A.constant rows + row) * A.constant shared
        rightOffset = (broadcastBatchIndex rightBatch batchDims batch * A.constant cols + col) * A.constant shared
     in multiply (left A.! A.index1 (leftOffset + inner)) (packedRight A.! A.index1 (rightOffset + inner))
  where
    packedRight = packColumns shared cols right

-- | Sum pairwise products of equally sized vectors.
dot :: (Shape '[n], DType a, Num (TensorElement a)) => Tensor '[n] a -> Tensor '[n] a -> Tensor '[] a
dot x y = Reduction.sum (zipWith (*) x y)

-- | Multiply promoted matrices over their broadcast batch dimensions.
matmul ::
  forall lhs rhs a.
  (DType a, Num (TensorElement a), Shape lhs, Shape rhs, Shape (MatmulShape lhs rhs)) =>
  Tensor lhs a ->
  Tensor rhs a ->
  Tensor (MatmulShape lhs rhs) a
matmul (Tensor left) (Tensor right) =
  Tensor $
    A.reshape (A.constant (backendShape (Proxy @(MatmulShape lhs rhs)))) $
      matrixProduct leftBatch rightBatch batchDims rows shared cols (A.flatten left) (A.flatten right)
  where
    (leftBatch, rows, shared) = matrixDimensions True (shape @lhs)
    (rightBatch, _, cols) = matrixDimensions False (shape @rhs)
    batchDims = broadcastDimensions leftBatch rightBatch

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
