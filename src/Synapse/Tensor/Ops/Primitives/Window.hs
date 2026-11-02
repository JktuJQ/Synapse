{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Shared sliding-window indexing and overlap accumulation.
module Synapse.Tensor.Ops.Primitives.Window (Window (..), window, coordinates, offset, windowLocation, extract, combine, patchesToMatrix, matrixToPatches, extractPatches, combinePatches) where

import qualified Data.Array.Accelerate as A
import Data.List (zipWith4)
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Ops.Primitives.Math (contract)
import Synapse.Tensor.Shapes.Dimensions (Shape (..), backendShape, strides)
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..))

-- | Spatial window geometry in channels-first order.
data Window = Window
  { batches :: Int,
    channels :: Int,
    inputSizes :: [Int],
    kernelSizes :: [Int],
    steps :: [Int],
    spacing :: [Int],
    beforePadding :: [Int],
    outputSizes :: [Int]
  }

-- | Reflect a validated channels-first input and its window parameters.
window :: [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> Window
window (batch : cin : spatial) kernel stride padding dilation = Window batch cin spatial kernel stride dilation before $ zipWith4 (\size k s (lo, hi) -> 1 + (size + lo + hi - k) `div` s) spatial effective stride pairs
  where
    effective = zipWith (\k d -> d * (k - 1) + 1) kernel dilation
    pairs = pair padding
    before = map fst pairs
    pair [] = []
    pair (lo : hi : rest) = (lo, hi) : pair rest
    pair _ = error "Invalid window padding rank"
window _ _ _ _ _ = error "Invalid window input rank"

-- | Decode row-major coordinates without constructing a runtime index list.
coordinates :: [Int] -> A.Exp Int -> [A.Exp Int]
coordinates dims linear = zipWith (\size stride -> linear `A.quot` A.constant stride `A.rem` A.constant size) dims $ strides dims

-- | Flatten row-major coordinates.
offset :: [Int] -> [A.Exp Int] -> A.Exp Int
offset dims = sum . zipWith (\stride coordinate -> A.constant stride * coordinate) (strides dims)

-- | Map a window column and flattened channel/kernel coordinate to a safe source offset.
windowLocation :: Window -> A.Exp Int -> A.Exp Int -> (A.Exp Bool, A.Exp Int)
windowLocation dims position inner = (valid, source)
  where
    spatial = product $ outputSizes dims
    volume = product $ kernelSizes dims
    b = position `A.quot` A.constant spatial
    c = inner `A.quot` A.constant volume
    output = coordinates (outputSizes dims) $ position `A.rem` A.constant spatial
    kernel = coordinates (kernelSizes dims) $ inner `A.rem` A.constant volume
    input = zipWith4 (\o s k p -> o * A.constant s + k - A.constant p) output (steps dims) (zipWith (\k d -> k * A.constant d) kernel $ spacing dims) (beforePadding dims)
    valid = foldr (A.&&) (A.constant True) $ zipWith (\size i -> i A.>= 0 A.&& i A.< A.constant size) (inputSizes dims) input
    safe = zipWith (\size i -> A.max 0 $ A.min (A.constant $ size - 1) i) (inputSizes dims) input
    source = (b * A.constant (channels dims) + c) * A.constant (product $ inputSizes dims) + offset (inputSizes dims) safe

-- | Extract a contraction-by-position matrix, using zero outside the input.
extract :: forall a. (DType a, Num (TensorElement a)) => Window -> A.Acc (A.Vector a) -> A.Acc (A.Matrix a)
extract dims input
  | 0 `elem` inputSizes dims = A.fill extent zero
  | otherwise = A.generate extent $ \ix ->
      let (valid, source) = windowLocation dims (A.indexHead ix) (A.indexHead $ A.indexTail ix)
       in A.cond valid (input A.! A.index1 source) zero
  where
    extent = A.constant $ A.Z A.:. (channels dims * product (kernelSizes dims)) A.:. (batches dims * product (outputSizes dims))
    zero = unTensorElement (0 :: TensorElement a)

-- | Sum all window entries referring to each original input element.
combine :: forall a. (DType a, Num (TensorElement a)) => Window -> A.Acc (A.Matrix a) -> A.Acc (A.Vector a)
combine dims patches = contract count volume $ \linear inner ->
  let plane = linear `A.quot` A.constant inputSpatial
      b = plane `A.quot` A.constant (channels dims)
      c = plane `A.rem` A.constant (channels dims)
      input = coordinates (inputSizes dims) $ linear `A.rem` A.constant inputSpatial
      kernel = coordinates (kernelSizes dims) inner
      shifted = zipWith4 (\i p k d -> i + A.constant p - k * A.constant d) input (beforePadding dims) kernel (spacing dims)
      output = zipWith (\i s -> i `A.quot` A.constant s) shifted (steps dims)
      valid = foldr (A.&&) (A.constant True) $ zipWith4 (\i s o size -> i A.>= 0 A.&& i `A.rem` A.constant s A.== 0 A.&& o A.< A.constant size) shifted (steps dims) output (outputSizes dims)
      safe = zipWith (\size o -> A.max 0 $ A.min (A.constant $ size - 1) o) (outputSizes dims) output
      position = b * A.constant outputSpatial + offset (outputSizes dims) safe
   in A.cond valid (patches A.! A.index2 (c * A.constant volume + inner) position) zero
  where
    inputSpatial = product $ inputSizes dims
    outputSpatial = product $ outputSizes dims
    count = batches dims * channels dims * inputSpatial
    volume = product $ kernelSizes dims
    zero = unTensorElement (0 :: TensorElement a)

-- | Restore the public batch/channel-window/position layout.
matrixToPatches :: (DType a) => Window -> A.Acc (A.Matrix a) -> A.Acc (A.Array A.DIM3 a)
matrixToPatches dims matrix = A.generate (A.constant $ A.Z A.:. batches dims A.:. (channels dims * product (kernelSizes dims)) A.:. spatial) $ \ix ->
  let b = A.indexHead $ A.indexTail $ A.indexTail ix
      k = A.indexHead $ A.indexTail ix
      p = A.indexHead ix
   in matrix A.! A.index2 k (b * A.constant spatial + p)
  where
    spatial = product $ outputSizes dims

-- | View public patches as a contraction-by-position matrix.
patchesToMatrix :: (DType a) => Window -> A.Acc (A.Vector a) -> A.Acc (A.Matrix a)
patchesToMatrix dims patches = A.generate (A.constant $ A.Z A.:. shared A.:. (batches dims * spatial)) $ \ix ->
  let k = A.indexHead $ A.indexTail ix
      position = A.indexHead ix
      b = position `A.quot` A.constant spatial
      p = position `A.rem` A.constant spatial
   in patches A.! A.index1 ((b * A.constant shared + k) * A.constant spatial + p)
  where
    spatial = product $ outputSizes dims
    shared = channels dims * product (kernelSizes dims)

-- | Extract row-major spatial patches without recording differentiation history.
extractPatches :: forall kernel stride padding dilation input out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, DType a, Num (TensorElement a)) => Tensor input a -> Tensor out a
extractPatches (Tensor input) = Tensor $ A.reshape (A.constant $ backendShape $ Proxy @out) $ matrixToPatches dims $ extract dims $ A.flatten input
  where
    dims = window (shape @input) (shape @kernel) (shape @stride) (shape @padding) (shape @dilation)

-- | Sum overlapping patches into a specified output shape.
combinePatches :: forall output kernel stride padding dilation patches a. (Shape output, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape patches, DType a, Num (TensorElement a)) => Tensor patches a -> Tensor output a
combinePatches (Tensor patches) = Tensor $ A.reshape (A.constant $ backendShape $ Proxy @output) $ combine dims $ patchesToMatrix dims $ A.flatten patches
  where
    dims = window (shape @output) (shape @kernel) (shape @stride) (shape @padding) (shape @dilation)
