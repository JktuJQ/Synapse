{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Reverse-mode rules for spatial convolutions and pooling.
module Synapse.Autograd.Tensor.Ops.Primitives.Convolution (convolution, convTranspose, maxPool, avgPool) where

import qualified Data.Array.Accelerate as A
import GHC.TypeNats (KnownNat)
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Primitives.Convolution (convolutionWindow, convolve, divisors, groupCount, kernelGradient, maxWindows, poolPatches, poolWindow, transposeConvolve)
import Synapse.Tensor.Ops.Primitives.Window (combine)
import Synapse.Tensor.Shapes.Dimensions (Shape (shape, shapeRank))
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..))

-- | Differentiate kernels and inputs through the same spatial window mapping.
convolution :: forall stride padding dilation groups input kernel out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape out, DType a, Num (TensorElement a)) => Tensor input a -> Tensor kernel a -> Tensor out a -> Tensor out a -> (Tensor input a, Tensor kernel a)
convolution (Tensor input) (Tensor weights) _ (Tensor seed) =
  ( Tensor $ A.reshape (A.shape input) $ transposeConvolve dims cout (groupCount @groups) (A.flatten weights) (A.flatten seed),
    Tensor $ A.reshape (A.shape weights) $ kernelGradient dims cout (groupCount @groups) (A.flatten input) (A.flatten seed)
  )
  where
    dims = convolutionWindow @input @kernel @stride @padding @dilation
    cout = product $ take 1 $ shape @kernel

-- | Differentiate a transposed convolution using its corresponding forward convolution.
convTranspose :: forall stride padding dilation groups input kernel out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape out, DType a, Num (TensorElement a)) => Tensor input a -> Tensor kernel a -> Tensor out a -> Tensor out a -> (Tensor input a, Tensor kernel a)
convTranspose (Tensor input) (Tensor weights) _ (Tensor seed) =
  ( Tensor $ A.reshape (A.shape input) $ convolve dims cout (groupCount @groups) (A.flatten seed) (A.flatten weights),
    Tensor $ A.reshape (A.shape weights) $ kernelGradient dims cout (groupCount @groups) (A.flatten seed) (A.flatten input)
  )
  where
    dims = convolutionWindow @out @kernel @stride @padding @dilation
    cout = shape @input !! 1

-- | Send each output seed to its selected input maximum, summing overlapping contributions.
maxPool :: forall kernel stride padding dilation input out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, DType a, Num (TensorElement a), E.Ord (TensorElement a)) => Tensor input a -> Tensor out a -> Tensor out a -> Tensor input a
maxPool (Tensor input) _ (Tensor seed) = Tensor $ A.reshape (A.shape input) $ combine dims $ poolPatches dims $ \row k ->
  A.cond (k A.== A.fst (selected A.! A.index1 row)) (gradient A.! A.index1 row) (unTensorElement (0 :: TensorElement a))
  where
    dims = poolWindow @kernel @stride @padding @input (shape @dilation)
    selected = maxWindows dims $ A.flatten input
    gradient = A.flatten seed

-- | Distribute each seed over its window using the forward divisor.
avgPool :: forall kernel stride padding input out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape out, DType a, Fractional (TensorElement a)) => Bool -> Tensor input a -> Tensor out a -> Tensor out a -> Tensor input a
avgPool includePadding (Tensor input) _ (Tensor seed) = Tensor $ A.reshape (A.shape input) $ combine dims $ poolPatches dims $ \row _ -> scaled A.! A.index1 row
  where
    dims = poolWindow @kernel @stride @padding @input (replicate (shapeRank @kernel) 1)
    scaled = A.zipWith (\x y -> unTensorElement $ TensorElement x / TensorElement y) (A.flatten seed) $ divisors includePadding dims
