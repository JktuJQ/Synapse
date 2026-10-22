{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Shape-changing tensor operations.
module Synapse.Tensor.Ops.Shape
  ( ShapeOps (..),
    broadcast,
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
    backendIndex,
    backendShape,
    shape,
    shapeRank,
    shapeSize,
    strides,
  )
import Synapse.Tensor.Shapes.Reshape (CanReshape)
import Synapse.Tensor.Type (Tensor (..))
import Prelude hiding (concat)

-- | Shape transformations with statically validated dimensions and axes.
class (DType a) => ShapeOps t a where
  -- | Reshape a tensor without changing its row-major element order.
  -- The source and target must have equal element counts, checked at compile time.
  reshapeOp ::
    forall to from.
    (Shape from, Shape to, CanReshape from to) =>
    t from a ->
    t to a

  -- | Remove a specified singleton axis.
  -- Axis numbering starts at zero on the left.
  -- Other singleton axes remain intact; non-singleton axes are rejected.
  squeezeOp ::
    forall axis sh.
    (Shape sh, Shape (RemoveAxis axis sh), CanSqueeze axis sh) =>
    Proxy axis ->
    t sh a ->
    t (RemoveAxis axis sh) a

  -- | Insert a singleton dimension.
  -- Positions range from zero through the rank, inclusive, so insertion before the first or after the last axis is permitted.
  unsqueezeOp ::
    forall axis sh.
    (Shape sh, Shape (InsertAxis axis 1 sh)) =>
    Proxy axis ->
    t sh a ->
    t (InsertAxis axis 1 sh) a

  -- | Reorder axes so each output axis corresponds to the specified input axis.
  -- For example, @permuteAxes \@'[1, 2, 0]@ maps @'[a, b, c]@ to @'[b, c, a]@.
  -- The order must contain every axis exactly once and scalars accept an empty order.
  permuteAxesOp ::
    forall xs sh.
    (Axes xs, Shape sh, Shape (PermuteAxes xs sh)) =>
    Proxy xs ->
    t sh a ->
    t (PermuteAxes xs sh) a

  -- | Transpose the last two axes of a matrix or batched matrix.
  -- Leading batch axes retain their order and inputs must have rank at least two.
  transposeOp ::
    forall sh.
    (Shape sh, Shape (TransposeShape sh)) =>
    t sh a ->
    t (TransposeShape sh) a

  -- | Concatenate two tensors along an existing zero-based axis.
  -- Both ranks and all dimensions outside that axis must match.
  concatOp ::
    forall axis lhs rhs.
    (KnownNat axis, Shape lhs, Shape rhs, Shape (ConcatShape axis lhs rhs)) =>
    Proxy axis ->
    t lhs a ->
    t rhs a ->
    t (ConcatShape axis lhs rhs) a

  -- | Stack two equally shaped tensors by inserting a dimension of size two.
  -- Positions range from zero through the input rank, so scalars may be stacked at position zero.
  stackOp ::
    forall axis sh.
    (KnownNat axis, Shape sh, Shape (InsertAxis axis 2 sh)) =>
    Proxy axis ->
    t sh a ->
    t sh a ->
    t (InsertAxis axis 2 sh) a

  -- | Broadcast a tensor to a target shape.
  --
  -- The target shape is usually inferred from context.
  -- When it is ambiguous, use type application:
  --
  -- @
  -- broadcast @'[32, 128] bias
  -- @
  broadcastOp ::
    forall to from.
    (Shape from, Shape to, CanBroadcast from to) =>
    t from a ->
    t to a

instance (DType a) => ShapeOps Tensor a where
  reshapeOp ::
    forall to from.
    (Shape from, Shape to, CanReshape from to) =>
    Tensor from a ->
    Tensor to a
  reshapeOp = reshapeTo @to

  squeezeOp ::
    forall axis sh.
    (Shape sh, Shape (RemoveAxis axis sh), CanSqueeze axis sh) =>
    Proxy axis ->
    Tensor sh a ->
    Tensor (RemoveAxis axis sh) a
  squeezeOp _ = reshapeTo @(RemoveAxis axis sh)

  unsqueezeOp ::
    forall axis sh.
    (Shape sh, Shape (InsertAxis axis 1 sh)) =>
    Proxy axis ->
    Tensor sh a ->
    Tensor (InsertAxis axis 1 sh) a
  unsqueezeOp _ = reshapeTo @(InsertAxis axis 1 sh)

  permuteAxesOp ::
    forall xs sh.
    (Axes xs, Shape sh, Shape (PermuteAxes xs sh)) =>
    Proxy xs ->
    Tensor sh a ->
    Tensor (PermuteAxes xs sh) a
  permuteAxesOp _ = reorderAxes @(PermuteAxes xs sh) (axes @xs)

  transposeOp ::
    forall sh.
    (Shape sh, Shape (TransposeShape sh)) =>
    Tensor sh a ->
    Tensor (TransposeShape sh) a
  transposeOp = reorderAxes @(TransposeShape sh) ([0 .. rank - 3] ++ [rank - 1, rank - 2])
    where
      rank = shapeRank @sh

  concatOp ::
    forall axis lhs rhs.
    (KnownNat axis, Shape lhs, Shape rhs, Shape (ConcatShape axis lhs rhs)) =>
    Proxy axis ->
    Tensor lhs a ->
    Tensor rhs a ->
    Tensor (ConcatShape axis lhs rhs) a
  concatOp _ = joinBlocks @(ConcatShape axis lhs rhs) outer leftWidth rightWidth
    where
      axis = axisPosition (Proxy @axis)
      leftDims = shape @lhs
      outer = product (take axis leftDims)
      leftWidth = product (drop axis leftDims)
      rightWidth = product (drop axis (shape @rhs))

  stackOp ::
    forall axis sh.
    (KnownNat axis, Shape sh, Shape (InsertAxis axis 2 sh)) =>
    Proxy axis ->
    Tensor sh a ->
    Tensor sh a ->
    Tensor (InsertAxis axis 2 sh) a
  stackOp _ = joinBlocks @(InsertAxis axis 2 sh) outer inner inner
    where
      axis = axisPosition (Proxy @axis)
      dims = shape @sh
      outer = product (take axis dims)
      inner = product (drop axis dims)

  broadcastOp ::
    forall to from.
    (Shape from, Shape to, CanBroadcast from to) =>
    Tensor from a ->
    Tensor to a
  broadcastOp (Tensor acc) =
    Tensor $
      A.backpermute
        (A.constant (backendShape (Proxy @to)))
        sourceIndex
        acc
    where
      sourceIndex :: A.Exp (ShapeToDIM to) -> A.Exp (ShapeToDIM from)
      sourceIndex ix = backendIndex @from (zipWith sourceCoordinate fromDims sourceCoordinates)
        where
          sourceCoordinates = drop leadingAxes (backendCoordinates @to ix)
          sourceCoordinate dim coord = if dim == 1 then 0 else coord

      fromDims = shape @from
      leadingAxes = shapeRank @to - shapeRank @from

-- | Reshape a tensor without changing its row-major element order.
-- The source and target must have equal element counts, checked at compile time.
reshape ::
  forall {t} to from a.
  (ShapeOps t a, Shape from, Shape to, CanReshape from to) =>
  t from a ->
  t to a
reshape = reshapeOp @t @a @to @from

-- | Remove a specified singleton axis.
-- Axis numbering starts at zero on the left.
-- Other singleton axes remain intact; non-singleton axes are rejected.
squeeze ::
  forall {t} axis sh a.
  (ShapeOps t a, Shape sh, Shape (RemoveAxis axis sh), CanSqueeze axis sh) =>
  t sh a ->
  t (RemoveAxis axis sh) a
squeeze = squeezeOp @t @a @axis @sh Proxy

-- | Insert a singleton dimension.
-- Positions range from zero through the rank, inclusive, so insertion before the first or after the last axis is permitted.
unsqueeze ::
  forall {t} axis sh a.
  (ShapeOps t a, Shape sh, Shape (InsertAxis axis 1 sh)) =>
  t sh a ->
  t (InsertAxis axis 1 sh) a
unsqueeze = unsqueezeOp @t @a @axis @sh Proxy

-- | Reorder axes so each output axis corresponds to the specified input axis.
-- For example, @permuteAxes \@'[1, 2, 0]@ maps @'[a, b, c]@ to @'[b, c, a]@.
-- The order must contain every axis exactly once and scalars accept an empty order.
permuteAxes ::
  forall {t} xs sh a.
  (ShapeOps t a, Axes xs, Shape sh, Shape (PermuteAxes xs sh)) =>
  t sh a ->
  t (PermuteAxes xs sh) a
permuteAxes = permuteAxesOp @t @a @xs @sh Proxy

-- | Transpose the last two axes of a matrix or batched matrix.
-- Leading batch axes retain their order and inputs must have rank at least two.
transpose ::
  forall {t} sh a.
  (ShapeOps t a, Shape sh, Shape (TransposeShape sh)) =>
  t sh a ->
  t (TransposeShape sh) a
transpose = transposeOp @t @a @sh

-- | Concatenate two tensors along an existing zero-based axis.
-- Both ranks and all dimensions outside that axis must match.
concat ::
  forall {t} axis lhs rhs a.
  (ShapeOps t a, KnownNat axis, Shape lhs, Shape rhs, Shape (ConcatShape axis lhs rhs)) =>
  t lhs a ->
  t rhs a ->
  t (ConcatShape axis lhs rhs) a
concat = concatOp @t @a @axis @lhs @rhs Proxy

-- | Stack two equally shaped tensors by inserting a dimension of size two.
-- Positions range from zero through the input rank, so scalars may be stacked at position zero.
stack ::
  forall {t} axis sh a.
  (ShapeOps t a, KnownNat axis, Shape sh, Shape (InsertAxis axis 2 sh)) =>
  t sh a ->
  t sh a ->
  t (InsertAxis axis 2 sh) a
stack = stackOp @t @a @axis @sh Proxy

-- | Broadcast a tensor to a target shape.
--
-- The target shape is usually inferred from context.
-- When it is ambiguous, use type application:
--
-- @
-- broadcast @'[32, 128] bias
-- @
broadcast ::
  forall {t} to from a.
  (ShapeOps t a, Shape from, Shape to, CanBroadcast from to) =>
  t from a ->
  t to a
broadcast = broadcastOp @t @a @to @from

-- | Change the shape without changing element order.
-- Callers must establish equal element counts before using this internal implementation.
reshapeTo ::
  forall to from a.
  (Shape from, Shape to, DType a) =>
  Tensor from a ->
  Tensor to a
reshapeTo (Tensor acc) = Tensor $ A.reshape (A.constant (backendShape (Proxy @to))) acc

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
