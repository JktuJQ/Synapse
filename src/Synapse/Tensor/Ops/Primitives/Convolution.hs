{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Spatial convolutions and pooling through shared window mappings.
module Synapse.Tensor.Ops.Primitives.Convolution (convolve, transposeConvolve, kernelGradient, convolutionWindow, groupCount, convolution, convTranspose, poolWindow, poolLocation, maxWindows, divisors, poolPatches, maxPool, avgPool) where

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Representation.Type as R
import qualified Data.Array.Accelerate.Sugar.Elt as Sugar
import qualified Data.Array.Accelerate.Type as AT
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat, natVal)
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Primitives.Math (contract, matrixProduct)
import Synapse.Tensor.Ops.Primitives.Window (Window (..), combine, extract, window, windowLocation)
import Synapse.Tensor.Shapes.Dimensions (Shape (..), backendShape)
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..))

-- | Reflect the validated geometry of a pooling operation.
poolWindow :: forall kernel stride padding input. (Shape kernel, Shape stride, Shape padding, Shape input) => [Int] -> Window
poolWindow = window (shape @input) (shape @kernel) (shape @stride) (shape @padding)

-- | Locate one kernel entry for a channels-first output element.
poolLocation :: Window -> A.Exp Int -> A.Exp Int -> (A.Exp Bool, A.Exp Int)
poolLocation dims row k = windowLocation dims (b * A.constant spatial + p) (c * A.constant volume + k)
  where
    spatial = product $ outputSizes dims
    volume = product $ kernelSizes dims
    b = row `A.quot` A.constant (channels dims * spatial)
    c = (row `A.quot` A.constant spatial) `A.rem` A.constant (channels dims)
    p = row `A.rem` A.constant spatial

-- | Padding contributes negative infinity for floats and the lower bound for integral elements.
paddingMinimum :: forall a. (DType a) => A.Exp a
paddingMinimum = A.constant $ Sugar.toElt @a $ lower $ Sugar.eltR @a
  where
    lower :: R.TypeR r -> r
    lower R.TupRunit = ()
    lower (R.TupRpair x y) = (lower x, lower y)
    lower (R.TupRsingle (AT.SingleScalarType (AT.NumSingleType (AT.IntegralNumType t)))) = case AT.integralDict t of
      AT.IntegralDict -> minBound
    lower (R.TupRsingle (AT.SingleScalarType (AT.NumSingleType (AT.FloatingNumType t)))) = case AT.floatingDict t of
      AT.FloatingDict -> -(1 / 0)
    lower (R.TupRsingle (AT.VectorScalarType _)) = error "Max pooling requires scalar numeric elements"

-- | Select the first row-major maximum, or the first NaN, from valid input positions.
maxWindows :: forall a. (DType a, Num (TensorElement a), E.Ord (TensorElement a)) => Window -> A.Acc (A.Vector a) -> A.Acc (A.Vector (Int, a))
maxWindows dims input = A.map discardValidity $ A.fold1 choose $ A.generate extent $ \ix ->
  let k = A.indexHead ix
      (valid, source) = poolLocation dims (A.indexHead $ A.indexTail ix) k
   in A.lift (valid, k, A.cond valid (input A.! A.index1 source) zero)
  where
    extent = A.constant $ A.Z A.:. (batches dims * channels dims * product (outputSizes dims)) A.:. product (kernelSizes dims)
    zero = paddingMinimum @a
    discardValidity value = let (valid, k, x) = A.unlift value :: (A.Exp Bool, A.Exp Int, A.Exp a) in A.lift (A.cond valid k (-1), x)
    choose lhs rhs =
      let (lv, li, lx) = A.unlift lhs :: (A.Exp Bool, A.Exp Int, A.Exp a)
          (rv, ri, rx) = A.unlift rhs :: (A.Exp Bool, A.Exp Int, A.Exp a)
          nanL = unTensorElement $ TensorElement lx E./= TensorElement lx
          nanR = unTensorElement $ TensorElement rx E./= TensorElement rx
          greater = unTensorElement $ TensorElement lx E.> TensorElement rx
          equal = unTensorElement $ TensorElement lx E.== TensorElement rx
          better = nanL A.&& (A.not nanR A.|| li A.<= ri) A.|| A.not nanL A.&& A.not nanR A.&& (greater A.|| equal A.&& li A.<= ri)
       in A.cond (lv A.&& (A.not rv A.|| better)) lhs rhs

-- | Count the divisor for each output window in the tensor's numeric type.
divisors :: forall a. (DType a, Num (TensorElement a)) => Bool -> Window -> A.Acc (A.Vector a)
divisors includePadding dims
  | includePadding = A.fill (A.constant $ A.Z A.:. count) $ unTensorElement (fromIntegral volume :: TensorElement a)
  | otherwise = contract count volume $ \row k -> A.cond (fst $ poolLocation dims row k) (unTensorElement (1 :: TensorElement a)) (unTensorElement (0 :: TensorElement a))
  where
    count = batches dims * channels dims * product (outputSizes dims)
    volume = product $ kernelSizes dims

-- | Arrange per-output kernel contributions for overlap accumulation.
poolPatches :: (DType a) => Window -> (A.Exp Int -> A.Exp Int -> A.Exp a) -> A.Acc (A.Matrix a)
poolPatches dims contribution = A.generate (A.constant $ A.Z A.:. (channels dims * volume) A.:. (batches dims * spatial)) $ \ix ->
  let inner = A.indexHead $ A.indexTail ix
      position = A.indexHead ix
      b = position `A.quot` A.constant spatial
      c = inner `A.quot` A.constant volume
      p = position `A.rem` A.constant spatial
      k = inner `A.rem` A.constant volume
   in contribution ((b * A.constant (channels dims) + c) * A.constant spatial + p) k
  where
    spatial = product $ outputSizes dims
    volume = product $ kernelSizes dims

-- | Reflect the window geometry of a channels-first convolution.
convolutionWindow :: forall input kernel stride padding dilation. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation) => Window
convolutionWindow = window (shape @input) (drop 2 $ shape @kernel) (shape @stride) (shape @padding) (shape @dilation)

-- | Reflect the positive, statically checked group count.
groupCount :: forall groups. (KnownNat groups) => Int
groupCount = fromIntegral $ natVal $ Proxy @groups

-- | Transpose each group matrix without expanding channels across groups.
transposeMatrix :: (DType a) => Int -> Int -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
transposeMatrix groups rows cols input = A.flatten $ A.backpermute (A.constant $ A.Z A.:. groups A.:. cols A.:. rows) source $ A.reshape (A.constant $ A.Z A.:. groups A.:. rows A.:. cols) input
  where
    source ix = A.index3 (A.indexHead $ A.indexTail $ A.indexTail ix) (A.indexHead ix) (A.indexHead $ A.indexTail ix)

-- | Reorder channels-first output into an output-channel-by-position matrix.
outputMatrix :: (DType a) => Window -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
outputMatrix dims cout input = A.generate (A.constant $ A.Z A.:. (cout * batches dims * spatial)) $ \ix ->
  let linear = A.unindex1 ix
      o = linear `A.quot` A.constant (batches dims * spatial)
      position = linear `A.rem` A.constant (batches dims * spatial)
      b = position `A.quot` A.constant spatial
      p = position `A.rem` A.constant spatial
   in input A.! A.index1 ((b * A.constant cout + o) * A.constant spatial + p)
  where
    spatial = product $ outputSizes dims

-- | Multiply kernels by windows and restore channels-first output order.
convolve :: (DType a, Num (TensorElement a)) => Window -> Int -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
convolve dims cout groups input weights = A.generate (A.constant $ A.Z A.:. (batches dims * cout * spatial)) $ \ix ->
  let linear = A.unindex1 ix
      position = (linear `A.quot` A.constant (cout * spatial)) * A.constant spatial + linear `A.rem` A.constant spatial
      o = (linear `A.quot` A.constant spatial) `A.rem` A.constant cout
   in result A.! A.index1 (o * A.constant positions + position)
  where
    spatial = product $ outputSizes dims
    positions = batches dims * spatial
    shared = channels dims `div` groups * product (kernelSizes dims)
    result = matrixProduct [groups] [groups] [groups] (cout `div` groups) shared positions weights $ A.flatten $ extract dims input

-- | Apply the transpose of a convolution's linear input mapping.
transposeConvolve :: (DType a, Num (TensorElement a)) => Window -> Int -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
transposeConvolve dims cout groups weights input = combine dims $ A.reshape (A.constant $ A.Z A.:. (groups * shared) A.:. positions) patches
  where
    shared = channels dims `div` groups * product (kernelSizes dims)
    positions = batches dims * product (outputSizes dims)
    patches = matrixProduct [groups] [groups] [groups] shared (cout `div` groups) positions (transposeMatrix groups (cout `div` groups) shared weights) (outputMatrix dims cout input)

-- | Contract an input's windows with channels-first output contributions.
kernelGradient :: (DType a, Num (TensorElement a)) => Window -> Int -> Int -> A.Acc (A.Vector a) -> A.Acc (A.Vector a) -> A.Acc (A.Vector a)
kernelGradient dims cout groups input seed = matrixProduct [groups] [groups] [groups] (cout `div` groups) positions shared (outputMatrix dims cout seed) $ transposeMatrix groups shared positions $ A.flatten $ extract dims input
  where
    shared = channels dims `div` groups * product (kernelSizes dims)
    positions = batches dims * product (outputSizes dims)

-- | Apply grouped cross-correlation with the validated output shape.
convolution :: forall stride padding dilation groups input kernel out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape out, DType a, Num (TensorElement a)) => Tensor input a -> Tensor kernel a -> Tensor out a
convolution (Tensor input) (Tensor weights) = Tensor $ A.reshape (A.constant $ backendShape $ Proxy @out) $ convolve dims (product $ take 1 $ shape @kernel) (groupCount @groups) (A.flatten input) (A.flatten weights)
  where
    dims = convolutionWindow @input @kernel @stride @padding @dilation

-- | Apply the transpose input mapping with the requested output shape.
convTranspose :: forall stride padding dilation groups input kernel out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape out, DType a, Num (TensorElement a)) => Tensor input a -> Tensor kernel a -> Tensor out a
convTranspose (Tensor input) (Tensor weights) = Tensor $ A.reshape (A.constant $ backendShape $ Proxy @out) $ transposeConvolve dims (shape @input !! 1) (groupCount @groups) (A.flatten weights) (A.flatten input)
  where
    dims = convolutionWindow @out @kernel @stride @padding @dilation

-- | Reduce each valid window to its maximum value.
maxPool :: forall kernel stride padding dilation input out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, DType a, Num (TensorElement a), E.Ord (TensorElement a)) => Tensor input a -> Tensor out a
maxPool (Tensor input) = Tensor $ A.reshape (A.constant $ backendShape $ Proxy @out) $ A.map A.snd $ maxWindows dims $ A.flatten input
  where
    dims = poolWindow @kernel @stride @padding @input (shape @dilation)

-- | Divide each window's sum by its full area or valid element count.
avgPool :: forall kernel stride padding input out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape out, DType a, Fractional (TensorElement a)) => Bool -> Tensor input a -> Tensor out a
avgPool includePadding (Tensor input) = Tensor $ A.reshape (A.constant $ backendShape $ Proxy @out) $ A.zipWith divide totals $ divisors includePadding dims
  where
    dims = poolWindow @kernel @stride @padding @input (replicate (shapeRank @kernel) 1)
    vector = A.flatten input
    totals = contract (batches dims * channels dims * product (outputSizes dims)) (product $ kernelSizes dims) $ \row k ->
      let (valid, source) = poolLocation dims row k
       in A.cond valid (vector A.! A.index1 source) (unTensorElement (0 :: TensorElement a))
    divide :: A.Exp a -> A.Exp a -> A.Exp a
    divide x y = unTensorElement $ TensorElement x / TensorElement y
