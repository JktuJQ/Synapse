{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Full and axis-wise reductions that remain inside the tensor computation.
module Synapse.Tensor.Ops.Reduction
  ( Reductions (..),
    MeanReductions (..),
    OrderedReductions (..),
    BooleanReductions (..),
    Normalization (..),
    LogReductions (..),
    sum,
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
import Synapse.Tensor.Ops.Elementwise (Ord (..), select, (&&), (||))
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Elementwise.Helpers (map, zipWith)
import Synapse.Tensor.Shapes.Axes (Axes (..), CanReduce, CanReduceNonEmpty, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendCoordinates, backendShape, shape, shapeSize, strides)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..), constant)
import Prelude hiding (Ord, all, any, map, max, maximum, min, minimum, sum, zipWith, (&&), (||))
import qualified Prelude as P

-- | Summation of full tensors and selected axes.
class (DType a) => Reductions t a where
  -- | Sum all elements, returning zero for an empty tensor.
  sumOp ::
    forall sh.
    (Shape sh) =>
    t sh a ->
    t '[] a

  -- | Sum over selected axes and remove them from the output shape.
  -- Axis order does not affect the result and an empty axis list preserves the input.
  -- Empty reduction regions produce zero.
  sumAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    t sh a ->
    t (ReduceShape xs sh) a

instance (DType a, Num (TensorElement a)) => Reductions Tensor a where
  sumOp (Tensor input) = Tensor (A.foldAll (elementOp (+)) (unTensorElement 0) input)

  sumAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    Tensor sh a ->
    Tensor (ReduceShape xs sh) a
  sumAxesOp _ input
    | reductionSize @xs @sh == 0 =
        Tensor (A.fill (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (unTensorElement 0))
    | otherwise = reduceAxesWith @xs (A.fold (elementOp (+)) (unTensorElement 0)) input

-- | Arithmetic means of full tensors and selected axes.
class (DType a) => MeanReductions t a where
  -- | Mean of all floating point elements, returning NaN for an empty tensor.
  meanOp ::
    forall sh.
    (Shape sh) =>
    t sh a ->
    t '[] a

  -- | Mean over selected axes, removing them from the output shape.
  -- Empty reduction regions produce NaN and an empty axis list preserves the input.
  meanAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    t sh a ->
    t (ReduceShape xs sh) a

instance (DType a, RealFloat a, Fractional (TensorElement a)) => MeanReductions Tensor a where
  meanOp ::
    forall sh.
    (Shape sh) =>
    Tensor sh a ->
    Tensor '[] a
  meanOp input = map (/ constant (fromIntegral (shapeSize @sh) :: a)) (sum input)

  meanAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    Tensor sh a ->
    Tensor (ReduceShape xs sh) a
  meanAxesOp _ = map (/ constant (fromIntegral (reductionSize @xs @sh) :: a)) . sumAxes @xs

-- | Extrema and their positions in tensors.
class (DType a) => OrderedReductions t a where
  -- | Minimum of all elements of a statically nonempty tensor.
  -- Floating point NaN values propagate to the result.
  minimumOp ::
    forall sh.
    (Shape sh, NonEmpty sh) =>
    t sh a ->
    t '[] a

  -- | Minimum over selected axes, removing them from the output shape.
  -- Selected dimensions must be nonzero, while unselected dimensions may be empty.
  -- Floating point NaN values propagate to the result.
  minimumAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh) =>
    Proxy xs ->
    t sh a ->
    t (ReduceShape xs sh) a

  -- | Maximum of all elements of a statically nonempty tensor.
  -- Floating point NaN values propagate to the result.
  maximumOp ::
    forall sh.
    (Shape sh, NonEmpty sh) =>
    t sh a ->
    t '[] a

  -- | Maximum over selected axes, removing them from the output shape.
  -- Selected dimensions must be nonzero, while unselected dimensions may be empty.
  -- Floating point NaN values propagate to the result.
  maximumAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh) =>
    Proxy xs ->
    t sh a ->
    t (ReduceShape xs sh) a

  -- | Position of the maximum along one nonempty axis, removing that axis.
  -- Positions start at zero; ties select the first occurrence and NaN selects the first NaN.
  argmaxOp ::
    forall axis sh.
    (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
    Proxy axis ->
    t sh a ->
    t (ReduceShape '[axis] sh) Int

  -- | Position of the minimum along one nonempty axis, removing that axis.
  -- Positions start at zero; ties select the first occurrence and NaN selects the first NaN.
  argminOp ::
    forall axis sh.
    (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
    Proxy axis ->
    t sh a ->
    t (ReduceShape '[axis] sh) Int

instance (DType a, Ord (TensorElement a)) => OrderedReductions Tensor a where
  minimumOp (Tensor input) = Tensor (A.fold1All (elementOp min) input)

  minimumAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh) =>
    Proxy xs ->
    Tensor sh a ->
    Tensor (ReduceShape xs sh) a
  minimumAxesOp _ = reduceAxesWith @xs (A.fold1 (elementOp min))

  maximumOp (Tensor input) = Tensor (A.fold1All (elementOp max) input)

  maximumAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh) =>
    Proxy xs ->
    Tensor sh a ->
    Tensor (ReduceShape xs sh) a
  maximumAxesOp _ = reduceAxesWith @xs (A.fold1 (elementOp max))

  argmaxOp ::
    forall axis sh.
    (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
    Proxy axis ->
    Tensor sh a ->
    Tensor (ReduceShape '[axis] sh) Int
  argmaxOp _ = argReduce @axis (E.>)

  argminOp ::
    forall axis sh.
    (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
    Proxy axis ->
    Tensor sh a ->
    Tensor (ReduceShape '[axis] sh) Int
  argminOp _ = argReduce @axis (E.<)

-- | Boolean conjunction and disjunction over tensor elements.
class BooleanReductions t where
  -- | Test whether every element is true, returning True for an empty tensor.
  allOp ::
    forall sh.
    (Shape sh) =>
    t sh Bool ->
    t '[] Bool

  -- | Test whether any element is true, returning False for an empty tensor.
  anyOp ::
    forall sh.
    (Shape sh) =>
    t sh Bool ->
    t '[] Bool

  -- | Test conjunction along selected axes, removing them from the result.
  -- Empty reduction regions produce True.
  allAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    t sh Bool ->
    t (ReduceShape xs sh) Bool

  -- | Test disjunction along selected axes, removing them from the result.
  -- Empty reduction regions produce False.
  anyAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    t sh Bool ->
    t (ReduceShape xs sh) Bool

instance BooleanReductions Tensor where
  allOp (Tensor input) = Tensor (A.foldAll (elementOp (&&)) (A.constant True) input)

  anyOp (Tensor input) = Tensor (A.foldAll (elementOp (||)) (A.constant False) input)

  allAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    Tensor sh Bool ->
    Tensor (ReduceShape xs sh) Bool
  allAxesOp _ = booleanAxes @xs True (&&)

  anyAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    Tensor sh Bool ->
    Tensor (ReduceShape xs sh) Bool
  anyAxesOp _ = booleanAxes @xs False (||)

-- | Stable normalization along a tensor axis.
class (DType a) => Normalization t a where
  -- | Normalize exponentials along one axis without overflowing on large finite inputs.
  -- The axis must be nonempty; NaN and infinite logits follow floating point arithmetic.
  softmaxOp ::
    forall axis sh.
    (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
    Proxy axis ->
    t sh a ->
    t sh a

  -- | Logarithm of softmax along one nonempty axis, computed without taking the log of small probabilities.
  logSoftmaxOp ::
    forall axis sh.
    (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
    Proxy axis ->
    t sh a ->
    t sh a

instance (DType a, Floating (TensorElement a), Ord (TensorElement a)) => Normalization Tensor a where
  softmaxOp ::
    forall axis sh.
    (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
    Proxy axis ->
    Tensor sh a ->
    Tensor sh a
  softmaxOp _ input = zipWith (/) exponentials (expandReduced @'[axis] @sh (sumAxes @'[axis] exponentials))
    where
      shifted = zipWith (-) input (expandReduced @'[axis] @sh (maximumAxes @'[axis] input))
      exponentials = map exp shifted

  logSoftmaxOp ::
    forall axis sh.
    (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
    Proxy axis ->
    Tensor sh a ->
    Tensor sh a
  logSoftmaxOp _ input = zipWith (-) shifted (expandReduced @'[axis] @sh (map log (sumAxes @'[axis] (map exp shifted))))
    where
      shifted = zipWith (-) input (expandReduced @'[axis] @sh (maximumAxes @'[axis] input))

-- | Stable logarithms of sums of exponentials.
class (DType a) => LogReductions t a where
  -- | Logarithm of the sum of exponentials, using maximum subtraction for finite inputs.
  -- Empty tensors produce negative infinity and NaN values propagate.
  logSumExpOp ::
    forall sh.
    (Shape sh) =>
    t sh a ->
    t '[] a

  -- | Logarithm of summed exponentials over selected axes, removing those axes.
  -- Empty regions produce negative infinity; infinite extrema are preserved and NaN values propagate.
  logSumExpAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    t sh a ->
    t (ReduceShape xs sh) a

instance (DType a, RealFloat a, Floating (TensorElement a), Ord (TensorElement a), E.RealFloat (TensorElement a)) => LogReductions Tensor a where
  logSumExpOp ::
    forall sh.
    (Shape sh) =>
    Tensor sh a ->
    Tensor '[] a
  logSumExpOp (Tensor input)
    | shapeSize @sh == 0 = Tensor (A.unit (A.constant (negate (1 / 0) :: a)))
    | otherwise = Tensor (A.unit (unTensorElement (select (E.isInfinite peak) peak (peak + log total))))
    where
      peak = TensorElement (A.the (A.fold1All (elementOp max) input))
      total = TensorElement (A.the (unTensor (sum (map (\x -> exp (x - peak)) (Tensor input :: Tensor sh a)))))

  logSumExpAxesOp ::
    forall xs sh.
    (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
    Proxy xs ->
    Tensor sh a ->
    Tensor (ReduceShape xs sh) a
  logSumExpAxesOp _ input
    | reductionSize @xs @sh == 0 = Tensor (A.fill (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (A.constant (negate (1 / 0) :: a)))
    | null (axes @xs) = Tensor (A.reshape (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (unTensor input))
    | otherwise = select (E.isInfinite peak) peak (zipWith (+) peak (map log total))
    where
      peak = reduceAxesWith @xs (A.fold1 (elementOp max)) input
      shifted = zipWith (-) input (expandReduced @xs @sh peak)
      total = sumAxes @xs (map exp shifted)

-- | Sum all elements, returning zero for an empty tensor.
sum ::
  forall {t} sh a.
  (Reductions t a, Shape sh) =>
  t sh a ->
  t '[] a
sum = sumOp @t @a @sh

-- | Sum over selected axes and remove them from the output shape.
-- Axis order does not affect the result and an empty axis list preserves the input.
-- Empty reduction regions produce zero.
sumAxes ::
  forall {t} xs sh a.
  (Reductions t a, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
sumAxes = sumAxesOp @t @a @xs @sh Proxy

-- | Mean of all floating point elements, returning NaN for an empty tensor.
mean ::
  forall {t} sh a.
  (MeanReductions t a, Shape sh) =>
  t sh a ->
  t '[] a
mean = meanOp @t @a @sh

-- | Mean over selected axes, removing them from the output shape.
-- Empty reduction regions produce NaN and an empty axis list preserves the input.
meanAxes ::
  forall {t} xs sh a.
  (MeanReductions t a, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
meanAxes = meanAxesOp @t @a @xs @sh Proxy

-- | Minimum of all elements of a statically nonempty tensor.
-- Floating point NaN values propagate to the result.
minimum ::
  forall {t} sh a.
  (OrderedReductions t a, Shape sh, NonEmpty sh) =>
  t sh a ->
  t '[] a
minimum = minimumOp @t @a @sh

-- | Minimum over selected axes, removing them from the output shape.
-- Selected dimensions must be nonzero, while unselected dimensions may be empty.
-- Floating point NaN values propagate to the result.
minimumAxes ::
  forall {t} xs sh a.
  (OrderedReductions t a, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
minimumAxes = minimumAxesOp @t @a @xs @sh Proxy

-- | Maximum of all elements of a statically nonempty tensor.
-- Floating point NaN values propagate to the result.
maximum ::
  forall {t} sh a.
  (OrderedReductions t a, Shape sh, NonEmpty sh) =>
  t sh a ->
  t '[] a
maximum = maximumOp @t @a @sh

-- | Maximum over selected axes, removing them from the output shape.
-- Selected dimensions must be nonzero, while unselected dimensions may be empty.
-- Floating point NaN values propagate to the result.
maximumAxes ::
  forall {t} xs sh a.
  (OrderedReductions t a, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
maximumAxes = maximumAxesOp @t @a @xs @sh Proxy

-- | Test whether every element is true, returning True for an empty tensor.
all ::
  forall {t} sh.
  (BooleanReductions t, Shape sh) =>
  t sh Bool ->
  t '[] Bool
all = allOp @t @sh

-- | Test whether any element is true, returning False for an empty tensor.
any ::
  forall {t} sh.
  (BooleanReductions t, Shape sh) =>
  t sh Bool ->
  t '[] Bool
any = anyOp @t @sh

-- | Test conjunction along selected axes, removing them from the result.
-- Empty reduction regions produce True.
allAxes ::
  forall {t} xs sh.
  (BooleanReductions t, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh Bool ->
  t (ReduceShape xs sh) Bool
allAxes = allAxesOp @t @xs @sh Proxy

-- | Test disjunction along selected axes, removing them from the result.
-- Empty reduction regions produce False.
anyAxes ::
  forall {t} xs sh.
  (BooleanReductions t, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh Bool ->
  t (ReduceShape xs sh) Bool
anyAxes = anyAxesOp @t @xs @sh Proxy

-- | Normalize exponentials along one axis without overflowing on large finite inputs.
-- The axis must be nonempty; NaN and infinite logits follow floating point arithmetic.
softmax ::
  forall {t} axis sh a.
  (Normalization t a, Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
  t sh a ->
  t sh a
softmax = softmaxOp @t @a @axis @sh Proxy

-- | Logarithm of softmax along one nonempty axis, computed without taking the log of small probabilities.
logSoftmax ::
  forall {t} axis sh a.
  (Normalization t a, Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
  t sh a ->
  t sh a
logSoftmax = logSoftmaxOp @t @a @axis @sh Proxy

-- | Logarithm of the sum of exponentials, using maximum subtraction for finite inputs.
-- Empty tensors produce negative infinity and NaN values propagate.
logSumExp ::
  forall {t} sh a.
  (LogReductions t a, Shape sh) =>
  t sh a ->
  t '[] a
logSumExp = logSumExpOp @t @a @sh

-- | Logarithm of summed exponentials over selected axes, removing those axes.
-- Empty regions produce negative infinity; infinite extrema are preserved and NaN values propagate.
logSumExpAxes ::
  forall {t} xs sh a.
  (LogReductions t a, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
logSumExpAxes = logSumExpAxesOp @t @a @xs @sh Proxy

-- | Position of the maximum along one nonempty axis, removing that axis.
-- Positions start at zero; ties select the first occurrence and NaN selects the first NaN.
argmax ::
  forall {t} axis sh a.
  (OrderedReductions t a, Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
  t sh a ->
  t (ReduceShape '[axis] sh) Int
argmax = argmaxOp @t @a @axis @sh Proxy

-- | Position of the minimum along one nonempty axis, removing that axis.
-- Positions start at zero; ties select the first occurrence and NaN selects the first NaN.
argmin ::
  forall {t} axis sh a.
  (OrderedReductions t a, Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
  t sh a ->
  t (ReduceShape '[axis] sh) Int
argmin = argminOp @t @a @axis @sh Proxy

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

-- | Choose an extremum together with its earliest position, including the first NaN.
-- Explicit position tie-breaking makes the result independent of the fold tree.
choosePosition :: (DType a, Ord (TensorElement a)) => (TensorElement a -> TensorElement a -> TensorElement Bool) -> A.Exp (Int, a) -> A.Exp (Int, a) -> A.Exp (Int, a)
choosePosition better left right = unTensorElement (select rightWins y x)
  where
    x = TensorElement left
    y = TensorElement right
    lx = TensorElement (A.snd left)
    ry = TensorElement (A.snd right)
    leftNaN = lx E./= lx
    rightNaN = ry E./= ry
    tied = (leftNaN && rightNaN) || (lx E.== ry)
    rightWins = (E.not leftNaN && (rightNaN || better ry lx)) || (tied && (TensorElement (A.fst right) E.< TensorElement (A.fst left)))

-- | Reduce one axis while retaining the position rather than the extremum value.
argReduce :: forall axis sh a. (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], DType a, Ord (TensorElement a)) => (TensorElement a -> TensorElement a -> TensorElement Bool) -> Tensor sh a -> Tensor (ReduceShape '[axis] sh) Int
argReduce better input
  | shapeSize @(ReduceShape '[axis] sh) == 0 = Tensor (A.use (A.fromList outputShape []))
  | otherwise = Tensor (A.reshape (A.constant outputShape) (A.map A.fst (A.fold1 (choosePosition better) positioned)))
  where
    outputShape = backendShape (Proxy @(ReduceShape '[axis] sh))
    positioned = A.imap (\ix value -> A.lift (A.indexHead ix, value)) (reductionMatrix @'[axis] input)
