{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Inner products and matrix multiplication.
module Synapse.Tensor.Ops.Raw.Math
    ( -- * Matrix products
      dot
    , matmulWith

      -- * Dimensions and batch broadcasting
    , matrixDimensions
    , broadcastDimensions
    , broadcastBatchIndex
    , broadcastCopies

      -- * Contraction helpers
    , multiply
    , contract
    , matrixProduct
    ) where

import Data.Array.Accelerate qualified as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Ops.Raw.Elementwise (zipWith)
import Synapse.Tensor.Ops.Raw.GEMM qualified as GEMM
import Synapse.Tensor.Ops.Raw.Reduction qualified as Reduction
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Backend (backendShape, strides)
import Synapse.Tensor.Shapes.Dimensions (Shape, shape)
import Prelude hiding (zipWith)
import Prelude qualified as P

-- | Sum pairwise products of equally sized vectors.
dot
    :: (Shape '[n], DType a, Num (TensorElement a), P.Num a)
    => Tensor '[n] a
    -> Tensor '[n] a
    -> Tensor '[] a
dot x y = Reduction.sum (zipWith (*) x y)

-- | Decode a row-major position into one coordinate per dimension.
coordinates
    :: [Int]
    -> A.Exp Int
    -> [A.Exp Int]
coordinates dims linear = P.zipWith (\dim stride -> linear `A.quot` A.constant stride `A.rem` A.constant dim) dims (strides dims)

-- | Promote a left vector to a row or a right vector to a column.
matrixDimensions
    :: Bool
    -> Bool
    -> [Int]
    -> ([Int], Int, Int)
matrixDimensions left transposed dims = case reverse dims of
    [features] -> if left then ([], 1, features) else ([], features, 1)
    cols : rows : batch
        | transposed -> (reverse batch, cols, rows)
        | otherwise -> (reverse batch, rows, cols)
    [] -> error "Synapse.Tensor.matmul: scalar operand"

-- | Combine trailing batch dimensions without materializing broadcast copies.
broadcastDimensions
    :: [Int]
    -> [Int]
    -> [Int]
broadcastDimensions lhs rhs = reverse (combine (reverse lhs) (reverse rhs))
  where
    combine [] ys = ys
    combine xs [] = xs
    combine (x : xs) (y : ys)
        | x == 1 = y : combine xs ys
        | y == 1 || x == y = x : combine xs ys
        | otherwise = error "Synapse.Tensor.matmul: incompatible batch dimensions"

-- | Project a common batch index onto an operand's original batch dimensions.
broadcastBatchIndex
    :: [Int]
    -> [Int]
    -> A.Exp Int
    -> A.Exp Int
broadcastBatchIndex source target linear
    | source == target = linear
    | null source = 0
    | otherwise =
        let padding = length target - length source
            sourceDims = replicate padding 1 ++ source
            sourceStrides = replicate padding 0 ++ strides source
         in sum
                [ coordinate * A.constant stride
                | (dim, stride, coordinate) <- zip3 sourceDims sourceStrides (coordinates target linear)
                , dim /= 1
                ]

-- | Enumerate the common batch positions contributing to one original batch position.
broadcastCopies
    :: [Int]
    -> [Int]
    -> (Int, A.Exp Int -> A.Exp Int -> A.Exp Int)
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
multiply
    :: Num (TensorElement a)
    => A.Exp a
    -> A.Exp a
    -> A.Exp a
multiply x y = unTensorElement (TensorElement x * TensorElement y)

-- | Sum one generated row of terms per output element, including empty contractions.
contract
    :: forall a
     . (DType a, Num (TensorElement a), P.Num a)
    => Int
    -> Int
    -> (A.Exp Int -> A.Exp Int -> A.Exp a)
    -> A.Acc (A.Vector a)
contract count width term
    | count == 0 = A.use (A.fromList (A.Z A.:. 0) [])
    | width == 0 = A.fill (A.constant (A.Z A.:. count)) zero
    | otherwise =
        A.fold (\x y -> unTensorElement (TensorElement x + TensorElement y)) zero $
            A.generate (A.constant (A.Z A.:. count A.:. width)) $ \ix ->
                term (A.indexHead (A.indexTail ix)) (A.indexHead ix)
  where
    zero = unTensorElement (0 :: TensorElement a)

-- | Transpose each matrix in a flattened row-major batch.
transposeMatrices
    :: DType a
    => Int
    -> Int
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
transposeMatrices rows cols input =
    A.generate (A.shape input) $ \ix ->
        let linear = A.unindex1 ix
            area = A.constant (rows * cols)
            batch = linear `A.quot` area
            position = linear `A.rem` area
            row = position `A.quot` A.constant rows
            col = position `A.rem` A.constant rows
         in input A.! A.index1 (batch * area + col * A.constant cols + row)

-- | Contract possibly transposed matrices across a broadcast batch.
matrixProduct
    :: (DType a, Num (TensorElement a), P.Num a)
    => [Int]
    -> [Int]
    -> [Int]
    -> Int
    -> Int
    -> Int
    -> Bool
    -> Bool
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
matrixProduct leftBatch rightBatch batchDims rows shared cols transposeLeft transposeRight left right
    | not transposeLeft && product rightBatch == 1 && leftBatch == batchDims =
        GEMM.matrixProduct 1 (batches * rows) shared cols 0 0 False transposeRight left right fallback
    | regular leftBatch && regular rightBatch =
        GEMM.matrixProduct
            batches
            rows
            shared
            cols
            leftStride
            rightStride
            transposeLeft
            transposeRight
            left
            right
            fallback
    | otherwise = fallback left right
  where
    batches = product batchDims
    regular dims = dims == batchDims || product dims == 1
    leftStride = if product leftBatch == 1 then 0 else rows * shared
    rightStride = if product rightBatch == 1 then 0 else shared * cols
    fallback lhs rhs = contract (batches * rows * cols) shared $ \linear inner ->
        let rowIndex = linear `A.quot` A.constant cols
            row = rowIndex `A.rem` A.constant rows
            batch = rowIndex `A.quot` A.constant rows
            col = linear `A.rem` A.constant cols
            leftOffset = (broadcastBatchIndex leftBatch batchDims batch * A.constant rows + row) * A.constant shared
            rightOffset = (broadcastBatchIndex rightBatch batchDims batch * A.constant cols + col) * A.constant shared
         in multiply (rowMajorLeft A.! A.index1 (leftOffset + inner)) (packedRight A.! A.index1 (rightOffset + inner))
      where
        rowMajorLeft = if transposeLeft then transposeMatrices shared rows lhs else lhs
        packedRight
            | transposeRight || cols <= 1 || shared == 0 = rhs
            | otherwise = A.compute $ transposeMatrices shared cols rhs

-- | Multiply vectors or matrices, broadcasting leading batch dimensions.
matmulWith
    :: forall lhs rhs out a
     . (DType a, Num (TensorElement a), Shape lhs, Shape rhs, Shape out, P.Num a)
    => Bool
    -> Bool
    -> Tensor lhs a
    -> Tensor rhs a
    -> Tensor out a
matmulWith transposeLeft transposeRight (Tensor left) (Tensor right) =
    Tensor $
        A.reshape (A.constant (backendShape (Proxy @out))) $
            matrixProduct
                leftBatch
                rightBatch
                batchDims
                rows
                shared
                cols
                transposeLeft
                transposeRight
                (A.flatten left)
                (A.flatten right)
  where
    (leftBatch, rows, shared) = matrixDimensions True transposeLeft (shape @lhs)
    (rightBatch, _, cols) = matrixDimensions False transposeRight (shape @rhs)
    batchDims = broadcastDimensions leftBatch rightBatch
