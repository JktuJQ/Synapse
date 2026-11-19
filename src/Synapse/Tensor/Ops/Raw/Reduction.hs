{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Sums, means, extrema, norms, and exponential normalization.
module Synapse.Tensor.Ops.Raw.Reduction
    ( -- * Sums and means
      sum
    , sumAxes
    , mean
    , meanAxes

      -- * Norms
    , normL1
    , normL2

      -- * Extrema and their positions
    , minimum
    , minimumAxes
    , argmin
    , maximum
    , maximumAxes
    , argmax

      -- * Boolean reductions
    , all
    , allAxes
    , any
    , anyAxes

      -- * Exponential reductions and normalization
    , logSumExp
    , logSumExpAxes
    , softmax
    , logSoftmax

      -- * Reduction geometry
    , reductionSize
    , expandReduced
    ) where

import qualified Data.Array.Accelerate as A
import Data.List (foldl', partition)
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Ops.Logic (Ord (..), select, (&&), (||))
import qualified Synapse.Tensor.Ops.Logic as E
import qualified Synapse.Tensor.Ops.Numeric as E
import Synapse.Tensor.Ops.Raw.Construction (element)
import Synapse.Tensor.Ops.Raw.Elementwise (map, zipWith)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Axes (Axes (..), CanReduce, CanReduceNonEmpty, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Backend (backendCoordinates, backendShape, strides)
import Synapse.Tensor.Shapes.Dimensions (Shape, shape, shapeSize)
import Prelude hiding (Ord, all, any, map, max, maximum, min, minimum, sum, zipWith, (&&), (||))
import qualified Prelude as P

-- | Apply an element operation to scalar expressions.
elementOp
    :: (TensorElement a -> TensorElement a -> TensorElement a)
    -> A.Exp a
    -> A.Exp a
    -> A.Exp a
elementOp f x y = unTensorElement (f (TensorElement x) (TensorElement y))

-- | Compute a source offset from grouped coordinates and their source strides.
sourceOffset
    :: A.Exp Int
    -> [(Int, Int, Int)]
    -> A.Exp Int
sourceOffset linear = foldl' addAxis 0
  where
    addAxis offset (dim, sourceStride, groupStride) =
        offset + ((linear `quot` A.constant groupStride) `rem` A.constant dim) * A.constant sourceStride

-- | Group retained axes into rows and reduced axes into columns in their original order.
reductionMatrix
    :: forall xs sh a
     . (Shape sh, Axes xs, DType a)
    => Tensor sh a
    -> A.Acc (A.Array A.DIM2 a)
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

-- | Apply one reduction after grouping all selected axes.
reduceAxesWith
    :: forall xs sh a
     . (Shape sh, Shape (ReduceShape xs sh), Axes xs, DType a)
    => (A.Acc (A.Array A.DIM2 a) -> A.Acc (A.Array A.DIM1 a))
    -> Tensor sh a
    -> Tensor (ReduceShape xs sh) a
reduceAxesWith reduce input
    | shapeSize @(ReduceShape xs sh) == 0 = Tensor (A.use (A.fromList outputShape []))
    | null (axes @xs) = Tensor (A.reshape (A.constant outputShape) (unTensor input))
    | otherwise = Tensor (A.reshape (A.constant outputShape) (reduce (reductionMatrix @xs input)))
  where
    outputShape = backendShape (Proxy @(ReduceShape xs sh))

-- | Sum elements.
sum
    :: (Shape sh, DType a, Num (TensorElement a), P.Num a)
    => Tensor sh a
    -> Tensor '[] a
sum (Tensor input) = Tensor (A.foldAll (elementOp (+)) (unTensorElement 0) input)

-- | Reduce boolean regions using the identity for empty regions.
booleanAxes
    :: forall xs sh
     . (Shape sh, Shape (ReduceShape xs sh), Axes xs)
    => Bool
    -> (TensorElement Bool -> TensorElement Bool -> TensorElement Bool)
    -> Tensor sh Bool
    -> Tensor (ReduceShape xs sh) Bool
booleanAxes identity combine input
    | reductionSize @xs @sh == 0 =
        Tensor (A.fill (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (A.constant identity))
    | otherwise = reduceAxesWith @xs (A.fold (elementOp combine) (A.constant identity)) input

-- | Repeat a reduced value over its original reduction region.
expandReduced
    :: forall xs sh a
     . (Shape sh, Shape (ReduceShape xs sh), Axes xs, DType a)
    => Tensor (ReduceShape xs sh) a
    -> Tensor sh a
expandReduced (Tensor input) = Tensor (A.backpermute (A.constant (backendShape (Proxy @sh))) offset (A.flatten input))
  where
    retained = [axis | axis <- [0 .. length (shape @sh) - 1], axis `notElem` axes @xs]
    outputStrides = strides [shape @sh !! axis | axis <- retained]
    offset ix =
        A.index1
            (foldl' (+) 0 (P.zipWith (\coordinate stride -> coordinate * A.constant stride) (coordinates ix) outputStrides))
    coordinates ix = [coordinate | (axis, coordinate) <- zip [0 ..] (backendCoordinates @sh ix), axis `notElem` axes @xs]

-- | Choose an extremum together with its earliest position, including the first NaN.
choosePosition
    :: (DType a, Ord (TensorElement a), P.Ord a)
    => (TensorElement a -> TensorElement a -> TensorElement Bool)
    -> A.Exp (Int, a)
    -> A.Exp (Int, a)
    -> A.Exp (Int, a)
choosePosition better left right = unTensorElement (select rightWins y x)
  where
    x = TensorElement left
    y = TensorElement right
    lx = TensorElement (A.snd left)
    ry = TensorElement (A.snd right)
    leftNaN = lx E./= lx
    rightNaN = ry E./= ry
    tied = (leftNaN && rightNaN) || (lx E.== ry)
    rightWins =
        (E.not leftNaN && (rightNaN || better ry lx))
            || (tied && (TensorElement (A.fst right) E.< TensorElement (A.fst left)))

-- | Reduce one axis while retaining the position rather than the extremum value.
argReduce
    :: forall axis sh a
     . (Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], DType a, Ord (TensorElement a), P.Ord a)
    => (TensorElement a -> TensorElement a -> TensorElement Bool)
    -> Tensor sh a
    -> Tensor (ReduceShape '[axis] sh) Int
argReduce better input
    | shapeSize @(ReduceShape '[axis] sh) == 0 = Tensor (A.use (A.fromList outputShape []))
    | otherwise =
        Tensor (A.reshape (A.constant outputShape) (A.map A.fst (A.fold1 (choosePosition better) positioned)))
  where
    outputShape = backendShape (Proxy @(ReduceShape '[axis] sh))
    positioned = A.imap (\ix value -> A.lift (A.indexHead ix, value)) (reductionMatrix @'[axis] input)

-- | Sum selected axes.
sumAxes
    :: forall xs sh a
     . (Shape sh, Shape (ReduceShape xs sh), Axes xs, DType a, Num (TensorElement a), P.Num a)
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
sumAxes input
    | reductionSize @xs @sh == 0 =
        Tensor (A.fill (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (unTensorElement 0))
    | otherwise = reduceAxesWith @xs (A.fold (elementOp (+)) (unTensorElement 0)) input

-- | Compute a full mean.
mean
    :: forall sh a
     . (Shape sh, DType a, RealFloat a, Fractional (TensorElement a))
    => Tensor sh a
    -> Tensor '[] a
mean input = map (/ element (fromIntegral (shapeSize @sh) :: a)) (sum input)

-- | Compute means over the selected axes.
meanAxes
    :: forall xs sh a
     . ( Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , DType a
       , RealFloat a
       , Fractional (TensorElement a)
       )
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
meanAxes = map (/ element (fromIntegral (reductionSize @xs @sh) :: a)) . sumAxes @xs

-- | Sum the absolute values of all elements.
normL1
    :: (Shape sh, DType a, Num (TensorElement a), P.Num a)
    => Tensor sh a
    -> Tensor '[] a
normL1 = sum . map abs

-- | Compute the square root of the sum of squared elements.
normL2
    :: forall sh a
     . ( Shape sh
       , DType a
       , Floating (TensorElement a)
       , Ord (TensorElement a)
       , E.RealFloat (TensorElement a)
       , P.Floating a
       , P.Ord a
       , P.RealFloat a
       )
    => Tensor sh a
    -> Tensor '[] a
normL2 input =
    let peak = TensorElement $ A.the $ A.foldAll (elementOp max) (unTensorElement 0) $ unTensor $ map abs input
        divisor = select ((peak E.== 0) || E.isInfinite peak) 1 peak
        squares = map (\x -> let scaled = x / divisor in scaled * scaled) input
        total = TensorElement $ A.the $ unTensor $ sum squares
        result = select (E.isInfinite peak) peak (peak * sqrt total)
     in Tensor $ A.unit $ unTensorElement result

-- | Minimum of all elements of a statically nonempty tensor.
minimum
    :: forall sh a
     . (DType a, Ord (TensorElement a), Shape sh, NonEmpty sh, P.Ord a)
    => Tensor sh a
    -> Tensor '[] a
minimum (Tensor input) = Tensor (A.fold1All (elementOp min) input)

-- | Minimum over selected axes, removing them from the output shape.
minimumAxes
    :: forall xs sh a
     . ( DType a
       , Ord (TensorElement a)
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduceNonEmpty xs sh
       , P.Ord a
       )
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
minimumAxes = reduceAxesWith @xs (A.fold1 (elementOp min))

-- | Position of the minimum along one nonempty axis, removing that axis.
argmin
    :: forall axis sh a
     . ( DType a
       , Ord (TensorElement a)
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , P.Ord a
       )
    => Tensor sh a
    -> Tensor (ReduceShape '[axis] sh) Int
argmin = argReduce @axis (E.<)

-- | Maximum of all elements of a statically nonempty tensor.
maximum
    :: forall sh a
     . (DType a, Ord (TensorElement a), Shape sh, NonEmpty sh, P.Ord a)
    => Tensor sh a
    -> Tensor '[] a
maximum (Tensor input) = Tensor (A.fold1All (elementOp max) input)

-- | Maximum over selected axes, removing them from the output shape.
maximumAxes
    :: forall xs sh a
     . ( DType a
       , Ord (TensorElement a)
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduceNonEmpty xs sh
       , P.Ord a
       )
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
maximumAxes = reduceAxesWith @xs (A.fold1 (elementOp max))

-- | Position of the maximum along one nonempty axis, removing that axis.
argmax
    :: forall axis sh a
     . ( DType a
       , Ord (TensorElement a)
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , P.Ord a
       )
    => Tensor sh a
    -> Tensor (ReduceShape '[axis] sh) Int
argmax = argReduce @axis (E.>)

-- | Test whether every element is true, returning True for an empty tensor.
all
    :: forall sh
     . Shape sh
    => Tensor sh Bool
    -> Tensor '[] Bool
all (Tensor input) = Tensor (A.foldAll (elementOp (&&)) (A.constant True) input)

-- | Test conjunction along selected axes, removing them from the result.
allAxes
    :: forall xs sh
     . (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => Tensor sh Bool
    -> Tensor (ReduceShape xs sh) Bool
allAxes = booleanAxes @xs True (&&)

-- | Test whether any element is true, returning False for an empty tensor.
any
    :: forall sh
     . Shape sh
    => Tensor sh Bool
    -> Tensor '[] Bool
any (Tensor input) = Tensor (A.foldAll (elementOp (||)) (A.constant False) input)

-- | Test disjunction along selected axes, removing them from the result.
anyAxes
    :: forall xs sh
     . (Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => Tensor sh Bool
    -> Tensor (ReduceShape xs sh) Bool
anyAxes = booleanAxes @xs False (||)

-- | Logarithm of the sum of exponentials, using maximum subtraction for finite inputs.
logSumExp
    :: forall sh a
     . ( DType a
       , RealFloat a
       , Floating (TensorElement a)
       , Ord (TensorElement a)
       , E.RealFloat (TensorElement a)
       , Shape sh
       )
    => Tensor sh a
    -> Tensor '[] a
logSumExp (Tensor input)
    | shapeSize @sh == 0 = Tensor (A.unit (A.constant (negate (1 / 0) :: a)))
    | otherwise = Tensor (A.unit (unTensorElement (select (E.isInfinite peak) peak (peak + log total))))
  where
    peak = TensorElement (A.the (A.fold1All (elementOp max) input))
    total = TensorElement (A.the (unTensor (sum (map (\x -> exp (x - peak)) (Tensor input :: Tensor sh a)))))

-- | Logarithm of summed exponentials over selected axes, removing those axes.
logSumExpAxes
    :: forall xs sh a
     . ( DType a
       , RealFloat a
       , Floating (TensorElement a)
       , Ord (TensorElement a)
       , E.RealFloat (TensorElement a)
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       )
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
logSumExpAxes input
    | reductionSize @xs @sh == 0 =
        Tensor (A.fill (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (A.constant (negate (1 / 0) :: a)))
    | null (axes @xs) =
        Tensor (A.reshape (A.constant (backendShape (Proxy @(ReduceShape xs sh)))) (unTensor input))
    | otherwise = select (map E.isInfinite peak) peak (zipWith (+) peak (map log total))
  where
    peak = reduceAxesWith @xs (A.fold1 (elementOp max)) input
    shifted = zipWith (-) input (expandReduced @xs @sh peak)
    total = sumAxes @xs (map exp shifted)

-- | Normalize exponentials along one axis without overflowing on large finite inputs.
softmax
    :: forall axis sh a
     . ( DType a
       , Floating (TensorElement a)
       , Ord (TensorElement a)
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , P.Floating a
       , P.Ord a
       )
    => Tensor sh a
    -> Tensor sh a
softmax input = zipWith (/) exponentials (expandReduced @'[axis] @sh (sumAxes @'[axis] exponentials))
  where
    shifted = zipWith (-) input (expandReduced @'[axis] @sh (maximumAxes @'[axis] input))
    exponentials = map exp shifted

-- | Logarithm of softmax along one nonempty axis, computed without taking the log of small probabilities.
logSoftmax
    :: forall axis sh a
     . ( DType a
       , Floating (TensorElement a)
       , Ord (TensorElement a)
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , P.Floating a
       , P.Ord a
       )
    => Tensor sh a
    -> Tensor sh a
logSoftmax input = zipWith (-) shifted (expandReduced @'[axis] @sh (map log (sumAxes @'[axis] (map exp shifted))))
  where
    shifted = zipWith (-) input (expandReduced @'[axis] @sh (maximumAxes @'[axis] input))

-- | Number of input elements contributing to each output element.
reductionSize
    :: forall xs sh
     . (Axes xs, Shape sh)
    => Int
reductionSize = product [dim | (axis, dim) <- zip [0 ..] (shape @sh), axis `elem` axes @xs]
