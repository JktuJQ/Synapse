{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Adjoint rules for extracting and combining spatial patches.
module Synapse.Autograd.Tensor.Ops.Primitives.Window (extractPatches, combinePatches) where

import qualified Data.Array.Accelerate as A
import qualified Synapse.Tensor.Ops.Primitives.Window as Raw
import Synapse.Tensor.Shapes.Dimensions (Shape (shape))
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement)

-- | Sum the seed's overlapping patches into the original input shape.
extractPatches :: forall kernel stride padding dilation input out a. (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, DType a, Num (TensorElement a)) => Tensor input a -> Tensor out a -> Tensor out a -> Tensor input a
extractPatches _ _ = Raw.combinePatches @input @kernel @stride @padding @dilation

-- | Extract output seed windows in the original patch layout.
combinePatches :: forall kernel stride padding dilation patches out a. (Shape patches, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape out, DType a, Num (TensorElement a)) => Tensor patches a -> Tensor out a -> Tensor out a -> Tensor patches a
combinePatches (Tensor patches) _ (Tensor seed) = Tensor $ A.reshape (A.shape patches) $ Raw.matrixToPatches dims $ Raw.extract dims $ A.flatten seed
  where
    dims = Raw.window (shape @out) (shape @kernel) (shape @stride) (shape @padding) (shape @dilation)
