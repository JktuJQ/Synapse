{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Normalization over trailing tensor dimensions.
module Synapse.NN.Layers.Normalization.LayerNorm (LayerNorm (..), layerNorm) where

import Control.Arrow (Arrow, arr)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat)
import Synapse.NN.Layers.Normalization.Statistics (expand, moments)
import Synapse.Tensor.Ops (TensorOps (..))
import qualified Synapse.Tensor.Ops.Shape as T
import Synapse.Tensor.Shapes.Axes (Axes, CanReduce, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Normalization (LayerNormAxes)
import Synapse.Tensor.Type (TensorElement, TensorLike, WithTensor)
import qualified Synapse.Tensor.Type as Tensor

-- | Per-element scale and bias over the trailing normalized dimensions.
-- Initialize weights to ones and bias to zeros.
data LayerNorm (normalized :: [Nat]) (t :: TensorLike) a = LayerNorm
  { -- | Scale for each normalized element.
    layerNormWeights :: t normalized a,
    -- | Offset for each normalized element.
    layerNormBias :: t normalized a
  }
  deriving (Generic)

type instance WithTensor t (LayerNorm normalized Tensor.Tensor a) = LayerNorm normalized t a

-- | Normalize trailing dimensions with population variance, then apply per-element scale and bias.
-- The normalized dimensions are inferred from the parameters and must match the input suffix.
-- Epsilon must be positive; training and evaluation use the same computation.
layerNorm ::
  forall normalized sh t a xs {arrow}.
  (Arrow arrow, xs ~ LayerNormAxes normalized sh, TensorOps t a, RealFloat a, Floating (TensorElement a), Floating (t sh a), Shape sh, Shape normalized, NonEmpty normalized, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh, CanBroadcast normalized sh) =>
  Double ->
  LayerNorm normalized t a ->
  arrow (t sh a) (t sh a)
layerNorm epsilon parameters = arr $ \input ->
  let (mean, variance) = moments @xs input
      normalized = (input - expand @xs input mean) / sqrt (expand @xs input variance + realToFrac epsilon)
   in normalized * T.broadcast (layerNormWeights parameters) + T.broadcast (layerNormBias parameters)
