{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Shape-changing tensor operations.
module Synapse.Tensor.Ops.Shape
  ( broadcast,
    reshape,
    squeeze,
    unsqueeze,
    transpose,
    permuteAxes,
    concat,
    stack,
  )
where

import qualified Data.Array.Accelerate as A
import Data.List (foldl')
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Axes (Axes (..), CanSqueeze, ConcatShape, InsertAxis, PermuteAxes, RemoveAxis, TransposeShape)
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions
  ( Shape,
    ShapeToDIM,
    backendCoordinates,
    backendShape,
    shape,
    shapeRank,
    shapeSize,
    strides,
  )
import Synapse.Tensor.Shapes.Reshape (CanReshape)
import Synapse.Tensor.Type (Tensor (..))
import Prelude hiding (concat)

-- | Change the shape without changing element order.
-- Callers must establish equal element counts before using this internal implementation.
reshapeTo ::
  forall to from a.
  (Shape from, Shape to, DType a) =>
  Tensor from a ->
  Tensor to a
reshapeTo (Tensor acc) = Tensor $ A.reshape (A.constant (backendShape (Proxy @to))) acc

-- | Reshape a tensor without changing its row-major element order.
-- The source and target must have equal element counts, checked at compile time.
reshape ::
  forall to from a.
  (Shape from, Shape to, CanReshape from to, DType a) =>
  Tensor from a ->
  Tensor to a
reshape = reshapeTo @to

-- | Remove a specified singleton axis.
-- Axis numbering starts at zero on the left.
-- Other singleton axes remain intact; non-singleton axes are rejected.
squeeze ::
  forall axis sh a.
  (Shape sh, Shape (RemoveAxis axis sh), CanSqueeze axis sh, DType a) =>
  Tensor sh a ->
  Tensor (RemoveAxis axis sh) a
squeeze = reshapeTo @(RemoveAxis axis sh)

-- | Insert a singleton dimension.
-- Positions range from zero through the rank, inclusive, so insertion before the first or after the last axis is permitted.
unsqueeze ::
  forall axis sh a.
  (Shape sh, Shape (InsertAxis axis 1 sh), DType a) =>
  Tensor sh a ->
  Tensor (InsertAxis axis 1 sh) a
unsqueeze = reshapeTo @(InsertAxis axis 1 sh)

-- | Apply a validated axis order using coordinates directly, without linear-index division.
reorderAxes :: forall to from a. (Shape from, Shape to, DType a) => [Int] -> Tensor from a -> Tensor to a
reorderAxes order tensor@(Tensor input)
  | order == [0 .. shapeRank @from - 1] = reshapeTo @to tensor
  | shapeSize @to == 0 = Tensor (A.use (A.fromList outputShape []))
  | otherwise = Tensor (A.backpermute (A.constant outputShape) sourceIndex (A.flatten input))
  where
    outputShape = backendShape (Proxy @to)
    sourceStrides = strides (shape @from)
    reorderedStrides = fmap (sourceStrides !!) order
    sourceIndex :: A.Exp (ShapeToDIM to) -> A.Exp A.DIM1
    sourceIndex ix = A.index1 $ foldl' addCoordinate 0 (zip (backendCoordinates @to ix) reorderedStrides)
    addCoordinate offset (coordinate, stride) = offset + coordinate * A.constant stride

-- | Reorder axes so each output axis corresponds to the specified input axis.
-- For example, @permuteAxes \@'[1, 2, 0]@ maps @'[a, b, c]@ to @'[b, c, a]@.
-- The order must contain every axis exactly once and scalars accept an empty order.
permuteAxes ::
  forall xs sh a.
  (Axes xs, Shape sh, Shape (PermuteAxes xs sh), DType a) =>
  Tensor sh a ->
  Tensor (PermuteAxes xs sh) a
permuteAxes = reorderAxes @(PermuteAxes xs sh) (axes @xs)

-- | Transpose the last two axes of a matrix or batched matrix.
-- Leading batch axes retain their order and inputs must have rank at least two.
transpose :: forall sh a. (Shape sh, Shape (TransposeShape sh), DType a) => Tensor sh a -> Tensor (TransposeShape sh) a
transpose = reorderAxes @(TransposeShape sh) ([0 .. rank - 3] ++ [rank - 1, rank - 2])
  where
    rank = shapeRank @sh

-- | Reflect a zero-based axis position for shape operations.
axisPosition :: (KnownNat axis) => proxy axis -> Int
axisPosition axis = fromInteger (toInteger (natVal axis))

-- | Join corresponding contiguous blocks through innermost matrix concatenation.
-- Empty blocks reuse the other input and completely empty outputs need no concatenation.
joinBlocks ::
  forall to lhs rhs a.
  (Shape lhs, Shape rhs, Shape to, DType a) =>
  Int ->
  Int ->
  Int ->
  Tensor lhs a ->
  Tensor rhs a ->
  Tensor to a
joinBlocks outer leftWidth rightWidth lhs@(Tensor left) rhs@(Tensor right)
  | shapeSize @to == 0 = Tensor (A.use (A.fromList outputShape []))
  | leftWidth == 0 = reshapeTo @to rhs
  | rightWidth == 0 = reshapeTo @to lhs
  | otherwise = Tensor $ A.reshape (A.constant outputShape) (leftMatrix A.++ rightMatrix)
  where
    outputShape = backendShape (Proxy @to)
    leftMatrix = A.reshape (A.constant (A.Z A.:. outer A.:. leftWidth)) left
    rightMatrix = A.reshape (A.constant (A.Z A.:. outer A.:. rightWidth)) right

-- | Concatenate two tensors along an existing zero-based axis.
-- Both ranks and all dimensions outside that axis must match.
concat ::
  forall axis lhs rhs a.
  (KnownNat axis, Shape lhs, Shape rhs, Shape (ConcatShape axis lhs rhs), DType a) =>
  Tensor lhs a ->
  Tensor rhs a ->
  Tensor (ConcatShape axis lhs rhs) a
concat = joinBlocks @(ConcatShape axis lhs rhs) outer leftWidth rightWidth
  where
    axis = axisPosition (Proxy @axis)
    leftDims = shape @lhs
    outer = product (take axis leftDims)
    leftWidth = product (drop axis leftDims)
    rightWidth = product (drop axis (shape @rhs))

-- | Stack two equally shaped tensors by inserting a dimension of size two.
-- Positions range from zero through the input rank, so scalars may be stacked at position zero.
stack ::
  forall axis sh a.
  (KnownNat axis, Shape sh, Shape (InsertAxis axis 2 sh), DType a) =>
  Tensor sh a ->
  Tensor sh a ->
  Tensor (InsertAxis axis 2 sh) a
stack = joinBlocks @(InsertAxis axis 2 sh) outer inner inner
  where
    axis = axisPosition (Proxy @axis)
    dims = shape @sh
    outer = product (take axis dims)
    inner = product (drop axis dims)

-- | Broadcast a tensor to a target shape.
--
-- The target shape is usually inferred from context.
-- When it is ambiguous, use type application:
--
-- @
-- broadcast @'[32, 128] bias
-- @
broadcast ::
  forall to from a.
  (Shape from, Shape to, CanBroadcast from to, DType a) =>
  Tensor from a ->
  Tensor to a
broadcast (Tensor acc) =
  Tensor $
    A.backpermute
      (A.constant (backendShape (Proxy @to)))
      sourceIndex
      acc
  where
    sourceIndex :: A.Exp (ShapeToDIM to) -> A.Exp (ShapeToDIM from)
    sourceIndex ix =
      A.fromIndex
        (A.constant (backendShape (Proxy @from)))
        (sourceLinearIndex ix)

    -- \| Map target indices to source elements; singleton source axes always contribute coordinate zero, and leading target axes do not contribute.
    sourceLinearIndex :: A.Exp (ShapeToDIM to) -> A.Exp Int
    sourceLinearIndex ix =
      foldr addAxis 0 axisMappings
      where
        targetLinear = A.toIndex (A.constant (backendShape (Proxy @to))) ix

        addAxis :: (Int, Int, (Int, Int)) -> A.Exp Int -> A.Exp Int
        addAxis (fromDim, toDim, (fromStride, toStride)) linearAcc
          | fromDim == 1 = linearAcc
          | otherwise =
              let coord = (targetLinear `quot` A.constant toStride) `rem` A.constant toDim
               in coord * A.constant fromStride + linearAcc

    fromDims = shape @from
    toDims = shape @to
    alignedToDims = drop (length toDims - length fromDims) toDims
    fromStrides = strides fromDims
    toStrides = strides toDims
    alignedToStrides = drop (length toStrides - length fromDims) toStrides
    axisMappings = zip3 fromDims alignedToDims (zip fromStrides alignedToStrides)
