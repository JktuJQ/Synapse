{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Coordinate transformations and contiguous tensor assembly.
module Synapse.Tensor.Primitives.Shape (reshape, permuteAxes, transpose, concat, stack, broadcast) where

import qualified Data.Array.Accelerate as A
import Data.List (foldl')
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.Shapes.Axes (Axes (..), ConcatShape, InsertAxis, PermuteAxes, TransposeShape)
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
import Synapse.Tensor.Type (DType, Tensor (..))
import Prelude hiding (concat)

-- | Change the shape without changing element order.
-- Callers must establish equal element counts before using this internal implementation.
reshape ::
  forall to from a.
  (Shape from, Shape to, DType a) =>
  Tensor from a ->
  Tensor to a
reshape (Tensor acc) = Tensor $ A.reshape (A.constant (backendShape (Proxy @to))) acc

-- | Apply a validated axis order using coordinates directly, without linear-index division.
reorderAxes :: forall to from a. (Shape from, Shape to, DType a) => [Int] -> Tensor from a -> Tensor to a
reorderAxes order tensor@(Tensor input)
  | order == [0 .. shapeRank @from - 1] = reshape @to tensor
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
  | leftWidth == 0 = reshape @to rhs
  | rightWidth == 0 = reshape @to lhs
  | otherwise = Tensor $ A.reshape (A.constant outputShape) (leftMatrix A.++ rightMatrix)
  where
    outputShape = backendShape (Proxy @to)
    leftMatrix = A.reshape (A.constant (A.Z A.:. outer A.:. leftWidth)) left
    rightMatrix = A.reshape (A.constant (A.Z A.:. outer A.:. rightWidth)) right

permuteAxes ::
  forall xs sh a.
  (DType a, Axes xs, Shape sh, Shape (PermuteAxes xs sh)) =>
  Proxy xs ->
  Tensor sh a ->
  Tensor (PermuteAxes xs sh) a
permuteAxes _ = reorderAxes @(PermuteAxes xs sh) (axes @xs)

transpose ::
  forall sh a.
  (DType a, Shape sh, Shape (TransposeShape sh)) =>
  Tensor sh a ->
  Tensor (TransposeShape sh) a
transpose = reorderAxes @(TransposeShape sh) ([0 .. rank - 3] ++ [rank - 1, rank - 2])
  where
    rank = shapeRank @sh

concat ::
  forall axis lhs rhs a.
  (DType a, KnownNat axis, Shape lhs, Shape rhs, Shape (ConcatShape axis lhs rhs)) =>
  Proxy axis ->
  Tensor lhs a ->
  Tensor rhs a ->
  Tensor (ConcatShape axis lhs rhs) a
concat _ = joinBlocks @(ConcatShape axis lhs rhs) outer leftWidth rightWidth
  where
    axis = axisPosition (Proxy @axis)
    leftDims = shape @lhs
    outer = product (take axis leftDims)
    leftWidth = product (drop axis leftDims)
    rightWidth = product (drop axis (shape @rhs))

stack ::
  forall axis sh a.
  (DType a, KnownNat axis, Shape sh, Shape (InsertAxis axis 2 sh)) =>
  Proxy axis ->
  Tensor sh a ->
  Tensor sh a ->
  Tensor (InsertAxis axis 2 sh) a
stack _ = joinBlocks @(InsertAxis axis 2 sh) outer inner inner
  where
    axis = axisPosition (Proxy @axis)
    dims = shape @sh
    outer = product (take axis dims)
    inner = product (drop axis dims)

broadcast ::
  forall to from a.
  (DType a, Shape from, Shape to, CanBroadcast from to) =>
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
    sourceIndex ix = backendIndex @from (zipWith sourceCoordinate fromDims sourceCoordinates)
      where
        sourceCoordinates = drop leadingAxes (backendCoordinates @to ix)
        sourceCoordinate dim coord = if dim == 1 then 0 else coord

    fromDims = shape @from
    leadingAxes = shapeRank @to - shapeRank @from
