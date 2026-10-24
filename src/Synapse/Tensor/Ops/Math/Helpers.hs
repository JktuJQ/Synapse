{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Internal matrix contractions and addressing of broadcast batches.
module Synapse.Tensor.Ops.Math.Helpers
  ( matrixDimensions,
    broadcastDimensions,
    broadcastBatchIndex,
    broadcastCopies,
    multiply,
    contract,
    matrixProduct,
  )
where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (strides)
import Synapse.Tensor.Type (TensorElement (..))

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
coordinates dims linear = zipWith (\dim stride -> linear `A.quot` A.constant stride `A.rem` A.constant dim) dims (strides dims)

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
       in sum (zipWith (*) (merge (zip [0 ..] fixed) varying) (map A.constant (strides target)))

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
