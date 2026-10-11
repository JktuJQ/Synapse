{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Full and axis-wise reductions that remain inside the tensor computation.
module Synapse.Tensor.Ops.Reduction
  ( sum,
    sumAxes,
    mean,
    meanAxes,
    minimum,
    minimumAxes,
    maximum,
    maximumAxes,
  )
where

import qualified Data.Array.Accelerate as A
import Data.List (foldl', partition)
import Data.Proxy (Proxy (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Elementwise (Ord (..), map)
import Synapse.Tensor.Shapes.Axes (Axes (..), CanReduce, CanReduceNonEmpty, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shape, shapeSize, strides)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..), constant)
import Prelude hiding (Ord, map, max, maximum, min, minimum, sum)

-- | Adapt an element operation to the backend fold representation.
elementOp :: (TensorElement a -> TensorElement a -> TensorElement a) -> A.Exp a -> A.Exp a -> A.Exp a
elementOp f x y = unTensorElement (f (TensorElement x) (TensorElement y))

-- | Compute a source offset from grouped coordinates and their source strides.
sourceOffset :: A.Exp Int -> [(Int, Int, Int)] -> A.Exp Int
sourceOffset linear = foldl' addAxis 0
  where
    addAxis offset (dim, sourceStride, groupStride) =
      offset + ((linear `quot` A.constant groupStride) `rem` A.constant dim) * A.constant sourceStride

-- | Group retained axes into rows and reduced axes into columns in their original order.
-- Already contiguous groups require only a reshape.
reductionMatrix :: forall xs sh a. (Shape sh, Axes xs, DType a) => Tensor sh a -> A.Acc (A.Array A.DIM2 a)
reductionMatrix (Tensor input)
  | mapAxis retained ++ mapAxis reduced == [0 .. length dims - 1] = A.reshape matrixShape input
  | otherwise = A.backpermute matrixShape sourceIndex input
  where
    dims = shape @sh
    selected = axes @xs
    entries = zip3 [0 :: Int ..] dims (strides dims)
    (reduced, retained) = partition (\(axis, _, _) -> axis `elem` selected) entries
    mapAxis = fmap (\(axis, _, _) -> axis)
    dimensions = fmap (\(_, dim, _) -> dim)
    mappings group = zipWith (\(_, dim, stride) groupStride -> (dim, stride, groupStride)) group (strides (dimensions group))
    matrixShape = A.constant (A.Z A.:. product (dimensions retained) A.:. product (dimensions reduced))
    sourceShape = A.constant (backendShape (Proxy @sh))
    retainedMappings = mappings retained
    reducedMappings = mappings reduced
    sourceIndex ix =
      A.fromIndex
        sourceShape
        (sourceOffset (A.indexHead (A.indexTail ix)) retainedMappings + sourceOffset (A.indexHead ix) reducedMappings)

-- | Apply one backend reduction after grouping all selected axes.
-- Empty output shapes are represented directly without executing a fold.
reduceAxesWith ::
  forall xs sh a.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, DType a) =>
  (A.Acc (A.Array A.DIM2 a) -> A.Acc (A.Array A.DIM1 a)) ->
  Tensor sh a ->
  Tensor (ReduceShape xs sh) a
reduceAxesWith reduce input
  | shapeSize @(ReduceShape xs sh) == 0 = Tensor (A.use (A.fromList outputShape []))
  | null (axes @xs) = Tensor (A.reshape (A.constant outputShape) (unTensor input))
  | otherwise = Tensor (A.reshape (A.constant outputShape) (reduce (reductionMatrix @xs input)))
  where
    outputShape = backendShape (Proxy @(ReduceShape xs sh))

-- | Number of input elements contributing to each output element.
reductionSize :: forall xs sh. (Axes xs, Shape sh) => Int
reductionSize = product [dim | (axis, dim) <- zip [0 ..] (shape @sh), axis `elem` axes @xs]

-- | Sum all elements, returning zero for an empty tensor.
sum :: (Shape sh, DType a, Num (TensorElement a)) => Tensor sh a -> TensorElement a
sum (Tensor input) = TensorElement (A.the (A.foldAll (elementOp (+)) (unTensorElement 0) input))

-- | Sum over selected axes and remove them from the output shape.
-- Axis order does not affect the result and an empty axis list preserves the input.
-- Empty reduction regions produce zero.
sumAxes ::
  forall xs sh a.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh, DType a, Num (TensorElement a)) =>
  Tensor sh a ->
  Tensor (ReduceShape xs sh) a
sumAxes input
  | reductionSize @xs @sh == 0 =
      Tensor (A.fill (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (unTensorElement 0))
  | otherwise = reduceAxesWith @xs (A.fold (elementOp (+)) (unTensorElement 0)) input

-- | Mean of all floating point elements, returning NaN for an empty tensor.
mean :: forall sh a. (Shape sh, DType a, RealFloat a, Fractional (TensorElement a)) => Tensor sh a -> TensorElement a
mean input = sum input / constant (fromIntegral (shapeSize @sh) :: a)

-- | Mean over selected axes, removing them from the output shape.
-- Empty reduction regions produce NaN and an empty axis list preserves the input.
meanAxes ::
  forall xs sh a.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh, DType a, RealFloat a, Fractional (TensorElement a)) =>
  Tensor sh a ->
  Tensor (ReduceShape xs sh) a
meanAxes = map (/ constant (fromIntegral (reductionSize @xs @sh) :: a)) . sumAxes @xs

-- | Minimum of all elements of a statically nonempty tensor.
-- Floating point NaN values propagate to the result.
minimum :: (Shape sh, NonEmpty sh, DType a, Ord (TensorElement a)) => Tensor sh a -> TensorElement a
minimum (Tensor input) = TensorElement (A.the (A.fold1All (elementOp min) input))

-- | Minimum over selected axes, removing them from the output shape.
-- Selected dimensions must be nonzero, while unselected dimensions may be empty.
-- Floating point NaN values propagate to the result.
minimumAxes ::
  forall xs sh a.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh, DType a, Ord (TensorElement a)) =>
  Tensor sh a ->
  Tensor (ReduceShape xs sh) a
minimumAxes = reduceAxesWith @xs (A.fold1 (elementOp min))

-- | Maximum of all elements of a statically nonempty tensor.
-- Floating point NaN values propagate to the result.
maximum :: (Shape sh, NonEmpty sh, DType a, Ord (TensorElement a)) => Tensor sh a -> TensorElement a
maximum (Tensor input) = TensorElement (A.the (A.fold1All (elementOp max) input))

-- | Maximum over selected axes, removing them from the output shape.
-- Selected dimensions must be nonzero, while unselected dimensions may be empty.
-- Floating point NaN values propagate to the result.
maximumAxes ::
  forall xs sh a.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh, DType a, Ord (TensorElement a)) =>
  Tensor sh a ->
  Tensor (ReduceShape xs sh) a
maximumAxes = reduceAxesWith @xs (A.fold1 (elementOp max))
