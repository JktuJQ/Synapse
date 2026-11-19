{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
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
import Synapse.Tensor.Ops.Raw.Window
    ( combine
    , convolutionWindow
    , convolve
    , divisors
    , groupCount
    , kernelGradient
    , maxWindows
    , poolPatches
    , poolWindow
    , transposeConvolve
    )
import Synapse.Tensor.Representation (Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Dimensions (Shape (shape, shapeRank))

import qualified Data.Array.Accelerate as A

import qualified Synapse.Tensor.Ops.Raw.Window as Raw

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
    -> Tensor out a
    -> Tensor out a
    -> (Tensor input a, Tensor kernel a)
convolution (Tensor input) (Tensor weights) _ (Tensor seed) =
    ( Tensor $
        A.reshape (A.shape input) $
            transposeConvolve dims cout (groupCount @groups) (A.flatten weights) (A.flatten seed)
    , Tensor $
        A.reshape (A.shape weights) $
            kernelGradient dims cout (groupCount @groups) (A.flatten input) (A.flatten seed)
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

-- | Send each output seed to its selected input maximum, summing overlapping contributions.
maxPool
    :: forall kernel stride padding dilation input out a
     . (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, NumDType a, OrdDType a)
    => Tensor input a
    -> Tensor out a
    -> Tensor out a
    -> Tensor input a
maxPool (Tensor input) _ (Tensor seed) = Tensor $ A.reshape (A.shape input) $ combine dims $ poolPatches dims $ \row k ->
    A.cond
        (k A.== A.fst (selected A.! A.index1 row))
        (gradient A.! A.index1 row)
        (unTensorElement (0 :: TensorElement a))
  where
    dims = poolWindow @kernel @stride @padding @input (shape @dilation)
    selected = maxWindows dims $ A.flatten input
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
