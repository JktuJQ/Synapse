{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

-- | Pullbacks for patch extraction, convolutions, and pooling.
module Synapse.Autograd.Tensor.Ops.Pullback.Window
    ( -- * Patches
      extractPatches
    , combinePatches

      -- * Convolutions
    , convolution
    , convTranspose

      -- * Pooling
    , maxPool
    , avgPool
    ) where

import GHC.TypeNats (KnownNat)

import Synapse.Tensor.Ops.Constraints (FractionalDType, NumDType, OrdDType)
import Synapse.Tensor.Ops.Raw.Math (matrixProduct)
import Synapse.Tensor.Ops.Raw.Window
    ( Window (..)
    , combine
    , convolutionWindow
    , convolve
    , divisors
    , groupCount
    , maxWindows
    , poolWindow
    , transposeConvolve
    )
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Dimensions (Shape (shape, shapeRank))

import Data.Array.Accelerate qualified as A

import Synapse.Tensor.Ops.Raw.Window qualified as Raw

-- | Sum the seed's overlapping patches into the original input shape.
extractPatches
    :: forall kernel stride padding dilation input out a
     . (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, NumDType a)
    => Tensor input a
    -> Tensor out a
    -> Tensor out a
    -> Tensor input a
extractPatches _ _ = Raw.combinePatches @input @kernel @stride @padding @dilation

-- | Extract output seed windows in the original patch layout.
combinePatches
    :: forall kernel stride padding dilation patches out a
     . (Shape patches, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, NumDType a)
    => Tensor patches a
    -> Tensor out a
    -> Tensor out a
    -> Tensor patches a
combinePatches (Tensor patches) _ (Tensor seed) = Tensor $ A.reshape (A.shape patches) $ Raw.matrixToPatches dims $ Raw.extract dims $ A.flatten seed
  where
    dims = Raw.window (shape @out) (shape @kernel) (shape @stride) (shape @padding) (shape @dilation)

-- | Contract an input's windows with channels-first output contributions.
kernelGradient
    :: (DType a, Num (TensorElement a), Num a)
    => Window
    -> Int
    -> Int
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
kernelGradient dims cout groups input =
    kernelGradientFromWindows dims cout groups (A.flatten $ Raw.extract dims input)

kernelGradientFromWindows
    :: (DType a, Num (TensorElement a), Num a)
    => Window
    -> Int
    -> Int
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
kernelGradientFromWindows dims cout groups windows seed =
    matrixProduct
        [groups]
        [groups]
        [groups]
        (cout `div` groups)
        positions
        shared
        (Raw.outputMatrix dims cout seed) $
        Raw.transposeMatrix groups shared positions windows
  where
    shared = channels dims `div` groups * product (kernelSizes dims)
    positions = batches dims * product (outputSizes dims)

-- | Differentiate kernels and inputs through the same spatial window mapping.
convolution
    :: forall stride padding dilation groups input kernel out a
     . ( Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape out
       , NumDType a
       )
    => Tensor input a
    -> Tensor kernel a
    -> A.Acc (A.Vector a)
    -> Tensor out a
    -> (Tensor input a, Tensor kernel a)
convolution (Tensor input) (Tensor weights) windows (Tensor seed) =
    ( Tensor $
        A.reshape (A.shape input) $
            transposeConvolve dims cout (groupCount @groups) (A.flatten weights) (A.flatten seed)
    , Tensor $
        A.reshape (A.shape weights) $
            kernelGradientFromWindows dims cout (groupCount @groups) windows (A.flatten seed)
    )
  where
    dims = convolutionWindow @input @kernel @stride @padding @dilation
    cout = product $ take 1 $ shape @kernel

-- | Differentiate a transposed convolution using its corresponding forward convolution.
convTranspose
    :: forall stride padding dilation groups input kernel out a
     . ( Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape out
       , NumDType a
       )
    => Tensor input a
    -> Tensor kernel a
    -> Tensor out a
    -> Tensor out a
    -> (Tensor input a, Tensor kernel a)
convTranspose (Tensor input) (Tensor weights) _ (Tensor seed) =
    ( Tensor $
        A.reshape (A.shape input) $
            convolve dims cout (groupCount @groups) (A.flatten seed) (A.flatten weights)
    , Tensor $
        A.reshape (A.shape weights) $
            kernelGradient dims cout (groupCount @groups) (A.flatten seed) (A.flatten input)
    )
  where
    dims = convolutionWindow @out @kernel @stride @padding @dilation
    cout = shape @input !! 1

-- | Arrange per-output kernel contributions for overlap accumulation.
poolPatches
    :: DType a
    => Window
    -> (A.Exp Int -> A.Exp Int -> A.Exp a)
    -> A.Acc (A.Matrix a)
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

-- | Send each output seed to its selected input maximum, summing overlapping contributions.
maxPool
    :: forall kernel stride padding dilation input out a
     . (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, NumDType a, OrdDType a)
    => Tensor input a
    -> Tensor out a
    -> Tensor out a
    -> Tensor input a
maxPool (Tensor input) _ (Tensor seed)
    | Raw.nonOverlapping dims =
        Tensor $ A.reshape (A.shape input) $ A.generate (A.shape $ A.flatten input) $ \ix ->
            let linear = A.unindex1 ix
                plane = linear `A.quot` A.constant (product $ Raw.inputSizes dims)
                coords = Raw.coordinates (Raw.inputSizes dims) (linear `A.rem` A.constant (product $ Raw.inputSizes dims))
                output = zipWith (\i size -> i `A.quot` A.constant size) coords (Raw.kernelSizes dims)
                kernel = zipWith (\i size -> i `A.rem` A.constant size) coords (Raw.kernelSizes dims)
                row = plane * A.constant (product $ Raw.outputSizes dims) + Raw.offset (Raw.outputSizes dims) output
                k = Raw.offset (Raw.kernelSizes dims) kernel
             in A.cond
                    (k A.== A.fst (selected A.! A.index1 row))
                    (gradient A.! A.index1 row)
                    (unTensorElement (0 :: TensorElement a))
    | otherwise = Tensor $ A.reshape (A.shape input) $ combine dims $ poolPatches dims $ \row k ->
        A.cond
            (k A.== A.fst (selected A.! A.index1 row))
            (gradient A.! A.index1 row)
            (unTensorElement (0 :: TensorElement a))
  where
    dims = poolWindow @kernel @stride @padding @input (shape @dilation)
    selected = A.compute $ maxWindows dims $ A.flatten input
    gradient = A.flatten seed

-- | Distribute each seed over its window using the forward divisor.
avgPool
    :: forall kernel stride padding input out a
     . (Shape input, Shape kernel, Shape stride, Shape padding, Shape out, FractionalDType a)
    => Bool
    -> Tensor input a
    -> Tensor out a
    -> Tensor out a
    -> Tensor input a
avgPool includePadding (Tensor input) _ (Tensor seed) = Tensor $ A.reshape (A.shape input) $ combine dims $ poolPatches dims $ \row _ -> scaled A.! A.index1 row
  where
    dims = poolWindow @kernel @stride @padding @input (replicate (shapeRank @kernel) 1)
    scaled =
        A.zipWith (\x y -> unTensorElement $ TensorElement x / TensorElement y) (A.flatten seed) $
            divisors includePadding dims
