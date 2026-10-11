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
    argmax,
    argmin,
    all,
    any,
    allAxes,
    anyAxes,
    softmax,
    logSoftmax,
    logSumExp,
    logSumExpAxes,
  )
where

import qualified Data.Array.Accelerate as A
import Data.List (foldl', partition)
import Data.Proxy (Proxy (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Elementwise (Ord (..), map, select, zipWith, (&&), (||))
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Shapes.Axes (Axes (..), CanReduce, CanReduceNonEmpty, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendCoordinates, backendShape, shape, shapeSize, strides)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..), constant)
import Prelude hiding (Ord, all, any, map, max, maximum, min, minimum, sum, zipWith, (&&), (||))
import qualified Prelude as P

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
reductionMatrix ::
  forall xs sh a.
  (Shape sh, Axes xs, DType a) =>
  Tensor sh a ->
  A.Acc (A.Array A.DIM2 a)
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
    mappings group = P.zipWith (\(_, dim, stride) groupStride -> (dim, stride, groupStride)) group (strides (dimensions group))
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
reductionSize ::
  forall xs sh.
  (Axes xs, Shape sh) =>
  Int
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
mean ::
  forall sh a.
  (Shape sh, DType a, RealFloat a, Fractional (TensorElement a)) =>
  Tensor sh a ->
  TensorElement a
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

-- | Test whether every element is true, returning True for an empty tensor.
all :: (Shape sh) => Tensor sh Bool -> TensorElement Bool
all (Tensor input) = TensorElement (A.the (A.foldAll (elementOp (&&)) (A.constant True) input))

-- | Test whether any element is true, returning False for an empty tensor.
any :: (Shape sh) => Tensor sh Bool -> TensorElement Bool
any (Tensor input) = TensorElement (A.the (A.foldAll (elementOp (||)) (A.constant False) input))

-- | Reduce boolean regions using the identity for empty regions.
booleanAxes ::
  forall xs sh.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs) =>
  Bool ->
  (TensorElement Bool -> TensorElement Bool -> TensorElement Bool) ->
  Tensor sh Bool ->
  Tensor (ReduceShape xs sh) Bool
booleanAxes identity combine input
  | reductionSize @xs @sh == 0 = Tensor (A.fill (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (A.constant identity))
  | otherwise = reduceAxesWith @xs (A.fold (elementOp combine) (A.constant identity)) input

-- | Test conjunction along selected axes, removing them from the result.
-- Empty reduction regions produce True.
allAxes ::
  forall xs sh.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  Tensor sh Bool ->
  Tensor (ReduceShape xs sh) Bool
allAxes = booleanAxes @xs True (&&)

-- | Test disjunction along selected axes, removing them from the result.
-- Empty reduction regions produce False.
anyAxes ::
  forall xs sh.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  Tensor sh Bool ->
  Tensor (ReduceShape xs sh) Bool
anyAxes = booleanAxes @xs False (||)

-- | Repeat a reduced value over its original reduction region.
expandReduced ::
  forall xs sh a.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, DType a) =>
  Tensor (ReduceShape xs sh) a ->
  Tensor sh a
expandReduced (Tensor input) = Tensor (A.backpermute (A.constant (backendShape (Proxy @sh))) offset (A.flatten input))
  where
    retained = [axis | axis <- [0 .. length (shape @sh) - 1], axis `notElem` axes @xs]
    outputStrides = strides [shape @sh !! axis | axis <- retained]
    offset ix = A.index1 (foldl' (+) 0 (P.zipWith (\coordinate stride -> coordinate * A.constant stride) (coordinates ix) outputStrides))
    coordinates ix = [coordinate | (axis, coordinate) <- zip [0 ..] (backendCoordinates @sh ix), axis `notElem` axes @xs]

-- | Normalize exponentials along one axis without overflowing on large finite inputs.
-- The axis must be nonempty; NaN and infinite logits follow floating point arithmetic.
softmax ::
  forall axis sh a.
  (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh, DType a, Floating (TensorElement a), Ord (TensorElement a)) =>
  Tensor sh a ->
  Tensor sh a
softmax input = zipWith (/) exponentials (expandReduced @'[axis] @sh (sumAxes @'[axis] exponentials))
  where
    shifted = zipWith (-) input (expandReduced @'[axis] @sh (maximumAxes @'[axis] input))
    exponentials = map exp shifted

-- | Logarithm of softmax along one nonempty axis, computed without taking the log of small probabilities.
logSoftmax ::
  forall axis sh a.
  (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh, DType a, Floating (TensorElement a), Ord (TensorElement a)) =>
  Tensor sh a ->
  Tensor sh a
logSoftmax input = zipWith (-) shifted (expandReduced @'[axis] @sh (map log (sumAxes @'[axis] (map exp shifted))))
  where
    shifted = zipWith (-) input (expandReduced @'[axis] @sh (maximumAxes @'[axis] input))

-- | Logarithm of the sum of exponentials, using maximum subtraction for finite inputs.
-- Empty tensors produce negative infinity and NaN values propagate.
logSumExp ::
  forall sh a.
  (Shape sh, DType a, RealFloat a, Floating (TensorElement a), Ord (TensorElement a), E.RealFloat (TensorElement a)) =>
  Tensor sh a ->
  TensorElement a
logSumExp (Tensor input)
  | shapeSize @sh == 0 = constant (negate (1 / 0) :: a)
  | otherwise = select (E.isInfinite peak) peak (peak + log total)
  where
    peak = TensorElement (A.the (A.fold1All (elementOp max) input))
    total = sum (map (\x -> exp (x - peak)) (Tensor input :: Tensor sh a))

-- | Logarithm of summed exponentials over selected axes, removing those axes.
-- Empty regions produce negative infinity; infinite extrema are preserved and NaN values propagate.
logSumExpAxes ::
  forall xs sh a.
  (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh, DType a, RealFloat a, Floating (TensorElement a), Ord (TensorElement a), E.RealFloat (TensorElement a)) =>
  Tensor sh a ->
  Tensor (ReduceShape xs sh) a
logSumExpAxes input
  | reductionSize @xs @sh == 0 = Tensor (A.fill (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (A.constant (negate (1 / 0) :: a)))
  | null (axes @xs) = Tensor (A.reshape (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (unTensor input))
  | otherwise = select (E.isInfinite peak) peak (zipWith (+) peak (map log total))
  where
    peak = reduceAxesWith @xs (A.fold1 (elementOp max)) input
    shifted = zipWith (-) input (expandReduced @xs @sh peak)
    total = sumAxes @xs (map exp shifted)

-- | Choose an extremum together with its earliest position, including the first NaN.
-- Explicit position tie-breaking makes the result independent of the fold tree.
choosePosition :: (DType a, Ord (TensorElement a)) => (TensorElement a -> TensorElement a -> TensorElement Bool) -> A.Exp (Int, a) -> A.Exp (Int, a) -> A.Exp (Int, a)
choosePosition better left right = unTensorElement (select rightWins y x)
  where
    x = TensorElement left
    y = TensorElement right
    lx = E.snd x
    ry = E.snd y
    leftNaN = lx E./= lx
    rightNaN = ry E./= ry
    tied = (leftNaN && rightNaN) || (lx E.== ry)
    rightWins = (E.not leftNaN && (rightNaN || better ry lx)) || (tied && (E.fst y E.< E.fst x))

-- | Reduce one axis while retaining the position rather than the extremum value.
argReduce :: forall axis sh a. (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], DType a, Ord (TensorElement a)) => (TensorElement a -> TensorElement a -> TensorElement Bool) -> Tensor sh a -> Tensor (ReduceShape '[axis] sh) Int
argReduce better input
  | shapeSize @(ReduceShape '[axis] sh) == 0 = Tensor (A.use (A.fromList outputShape []))
  | otherwise = Tensor (A.reshape (A.constant outputShape) (A.map A.fst (A.fold1 (choosePosition better) positioned)))
  where
    outputShape = backendShape (Proxy @(ReduceShape '[axis] sh))
    positioned = A.imap (\ix value -> A.lift (A.indexHead ix, value)) (reductionMatrix @'[axis] input)

-- | Position of the maximum along one nonempty axis, removing that axis.
-- Positions start at zero; ties select the first occurrence and NaN selects the first NaN.
argmax :: forall axis sh a. (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh, DType a, Ord (TensorElement a)) => Tensor sh a -> Tensor (ReduceShape '[axis] sh) Int
argmax = argReduce @axis (E.>)

-- | Position of the minimum along one nonempty axis, removing that axis.
-- Positions start at zero; ties select the first occurrence and NaN selects the first NaN.
argmin :: forall axis sh a. (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh, DType a, Ord (TensorElement a)) => Tensor sh a -> Tensor (ReduceShape '[axis] sh) Int
argmin = argReduce @axis (E.<)
