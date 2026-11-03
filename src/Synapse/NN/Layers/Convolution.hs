{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Channels-first convolution parameters and their functional application.
module Synapse.NN.Layers.Convolution (Convolution (..), TransposedConvolution (..), conv1d, conv2d, conv3d, convTranspose1d, convTranspose2d, convTranspose3d) where

import Control.Arrow (Arrow, arr)
import GHC.Generics (Generic)
import GHC.TypeLits (Div, KnownNat, Mod, Nat)
import Synapse.Tensor.Ops (TensorOps)
import qualified Synapse.Tensor.Ops.Convolution as T
import Synapse.Tensor.Ops.Shape (broadcast, reshape)
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Convolution (Conv1dShape, Conv2dShape, Conv3dShape, ConvTranspose1dShape, ConvTranspose2dShape, ConvTranspose3dShape)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (TensorLike, WithTensor)
import qualified Synapse.Tensor.Type as Tensor

-- | Weights and per-output-channel bias for an ordinary spatial convolution.
-- The kernel is a list of one, two or three spatial sizes; each filter consumes @in_channels / groups@ channels.
-- Stride, padding and dilation are supplied when applying the layer.
-- Initialize the fields explicitly, as with "Synapse.NN.Layers.Dense".
data Convolution (in_channels :: Nat) (out_channels :: Nat) (kernel :: [Nat]) (groups :: Nat) (t :: TensorLike) a = Convolution
  { -- | Weights in output-channel, input-channel-per-group, spatial order.
    convolutionWeights :: t (out_channels ': Div in_channels groups ': kernel) a,
    -- | Bias shared across batches and spatial positions.
    convolutionBias :: t '[out_channels] a
  }
  deriving (Generic)

type instance WithTensor t (Convolution cin cout kernel groups Tensor.Tensor a) = Convolution cin cout kernel groups t a

-- | Weights and bias for a transposed spatial convolution.
-- Each input channel has @out_channels / groups@ spatial filters; input and output channels are split into the same number of groups.
-- Output padding is supplied when applying the layer and selects among compatible output shapes.
data TransposedConvolution (in_channels :: Nat) (out_channels :: Nat) (kernel :: [Nat]) (groups :: Nat) (t :: TensorLike) a = TransposedConvolution
  { -- | Weights in input-channel, output-channel-per-group, spatial order.
    transposedWeights :: t (in_channels ': Div out_channels groups ': kernel) a,
    -- | Bias shared across batches and spatial positions.
    transposedBias :: t '[out_channels] a
  }
  deriving (Generic)

type instance WithTensor t (TransposedConvolution cin cout kernel groups Tensor.Tensor a) = TransposedConvolution cin cout kernel groups t a

-- | Apply a 1D convolution and add bias to every output channel in NCL layout.
-- Type arguments specify stride, @[left, right]@ padding and dilation.
-- Channel counts, kernel sizes and groups are inferred from the parameter record.
conv1d ::
  forall stride padding dilation {in_channels} {out_channels} {kernel} {groups} {batch} {length} {t} a out_sh {arrow}.
  ( Arrow arrow,
    TensorOps t a,
    Num (Tensor.TensorElement a),
    Num (t out_sh a),
    KnownNat out_channels,
    KnownNat groups,
    Shape stride,
    Shape padding,
    Shape dilation,
    Shape '[batch, in_channels, length],
    Shape (out_channels ': Div in_channels groups ': kernel),
    out_sh ~ Conv1dShape '[batch, in_channels, length] (out_channels ': Div in_channels groups ': kernel) stride padding dilation groups,
    Shape out_sh,
    CanBroadcast '[out_channels, 1] out_sh
  ) =>
  Convolution in_channels out_channels kernel groups t a ->
  arrow (t '[batch, in_channels, length] a) (t out_sh a)
conv1d parameters = arr $ \input -> T.conv1d @stride @padding @dilation @groups input (convolutionWeights parameters) + broadcast (reshape @'[out_channels, 1] (convolutionBias parameters))

-- | Apply a 2D convolution and add bias to every output channel in NCHW layout.
-- Type arguments specify stride, @[top, bottom, left, right]@ padding and dilation.
-- Channel counts, kernel sizes and groups are inferred from the parameter record.
conv2d ::
  forall stride padding dilation {in_channels} {out_channels} {kernel} {groups} {batch} {height} {width} {t} a out_sh {arrow}.
  ( Arrow arrow,
    TensorOps t a,
    Num (Tensor.TensorElement a),
    Num (t out_sh a),
    KnownNat out_channels,
    KnownNat groups,
    Shape stride,
    Shape padding,
    Shape dilation,
    Shape '[batch, in_channels, height, width],
    Shape (out_channels ': Div in_channels groups ': kernel),
    out_sh ~ Conv2dShape '[batch, in_channels, height, width] (out_channels ': Div in_channels groups ': kernel) stride padding dilation groups,
    Shape out_sh,
    CanBroadcast '[out_channels, 1, 1] out_sh
  ) =>
  Convolution in_channels out_channels kernel groups t a ->
  arrow (t '[batch, in_channels, height, width] a) (t out_sh a)
conv2d parameters = arr $ \input -> T.conv2d @stride @padding @dilation @groups input (convolutionWeights parameters) + broadcast (reshape @'[out_channels, 1, 1] (convolutionBias parameters))

-- | Apply a 3D convolution and add bias to every output channel in NCDHW layout.
-- Type arguments specify stride, @[front, back, top, bottom, left, right]@ padding and dilation.
-- Channel counts, kernel sizes and groups are inferred from the parameter record.
conv3d ::
  forall stride padding dilation {in_channels} {out_channels} {kernel} {groups} {batch} {depth} {height} {width} {t} a out_sh {arrow}.
  ( Arrow arrow,
    TensorOps t a,
    Num (Tensor.TensorElement a),
    Num (t out_sh a),
    KnownNat out_channels,
    KnownNat groups,
    Shape stride,
    Shape padding,
    Shape dilation,
    Shape '[batch, in_channels, depth, height, width],
    Shape (out_channels ': Div in_channels groups ': kernel),
    out_sh ~ Conv3dShape '[batch, in_channels, depth, height, width] (out_channels ': Div in_channels groups ': kernel) stride padding dilation groups,
    Shape out_sh,
    CanBroadcast '[out_channels, 1, 1, 1] out_sh
  ) =>
  Convolution in_channels out_channels kernel groups t a ->
  arrow (t '[batch, in_channels, depth, height, width] a) (t out_sh a)
conv3d parameters = arr $ \input -> T.conv3d @stride @padding @dilation @groups input (convolutionWeights parameters) + broadcast (reshape @'[out_channels, 1, 1, 1] (convolutionBias parameters))

-- | Apply a 1D transposed convolution and add bias to every output channel in NCL layout.
-- Type arguments specify stride, @[left, right]@ padding and dilation, then @[extra_length]@ output padding.
-- Channel counts, kernel sizes and groups are inferred from the parameter record.
convTranspose1d ::
  forall stride padding dilation extra {in_channels} {out_channels} {kernel} {groups} {batch} {length} {t} a out_sh {arrow}.
  ( Arrow arrow,
    TensorOps t a,
    Num (Tensor.TensorElement a),
    Num (t out_sh a),
    KnownNat out_channels,
    KnownNat groups,
    Mod out_channels groups ~ 0,
    Shape stride,
    Shape padding,
    Shape dilation,
    Shape '[batch, in_channels, length],
    Shape (in_channels ': Div out_channels groups ': kernel),
    out_sh ~ ConvTranspose1dShape '[batch, in_channels, length] (in_channels ': Div out_channels groups ': kernel) stride padding dilation groups extra,
    Shape out_sh,
    CanBroadcast '[out_channels, 1] out_sh
  ) =>
  TransposedConvolution in_channels out_channels kernel groups t a ->
  arrow (t '[batch, in_channels, length] a) (t out_sh a)
convTranspose1d parameters = arr $ \input -> T.convTranspose1d @stride @padding @dilation @groups @extra input (transposedWeights parameters) + broadcast (reshape @'[out_channels, 1] (transposedBias parameters))

-- | Apply a 2D transposed convolution and add bias to every output channel in NCHW layout.
-- Type arguments specify stride, @[top, bottom, left, right]@ padding and dilation, then @[extra_height, extra_width]@ output padding.
-- Channel counts, kernel sizes and groups are inferred from the parameter record.
convTranspose2d ::
  forall stride padding dilation extra {in_channels} {out_channels} {kernel} {groups} {batch} {height} {width} {t} a out_sh {arrow}.
  ( Arrow arrow,
    TensorOps t a,
    Num (Tensor.TensorElement a),
    Num (t out_sh a),
    KnownNat out_channels,
    KnownNat groups,
    Mod out_channels groups ~ 0,
    Shape stride,
    Shape padding,
    Shape dilation,
    Shape '[batch, in_channels, height, width],
    Shape (in_channels ': Div out_channels groups ': kernel),
    out_sh ~ ConvTranspose2dShape '[batch, in_channels, height, width] (in_channels ': Div out_channels groups ': kernel) stride padding dilation groups extra,
    Shape out_sh,
    CanBroadcast '[out_channels, 1, 1] out_sh
  ) =>
  TransposedConvolution in_channels out_channels kernel groups t a ->
  arrow (t '[batch, in_channels, height, width] a) (t out_sh a)
convTranspose2d parameters = arr $ \input -> T.convTranspose2d @stride @padding @dilation @groups @extra input (transposedWeights parameters) + broadcast (reshape @'[out_channels, 1, 1] (transposedBias parameters))

-- | Apply a 3D transposed convolution and add bias to every output channel in NCDHW layout.
-- Type arguments specify stride, @[front, back, top, bottom, left, right]@ padding and dilation, then @[extra_depth, extra_height, extra_width]@ output padding.
-- Channel counts, kernel sizes and groups are inferred from the parameter record.
convTranspose3d ::
  forall stride padding dilation extra {in_channels} {out_channels} {kernel} {groups} {batch} {depth} {height} {width} {t} a out_sh {arrow}.
  ( Arrow arrow,
    TensorOps t a,
    Num (Tensor.TensorElement a),
    Num (t out_sh a),
    KnownNat out_channels,
    KnownNat groups,
    Mod out_channels groups ~ 0,
    Shape stride,
    Shape padding,
    Shape dilation,
    Shape '[batch, in_channels, depth, height, width],
    Shape (in_channels ': Div out_channels groups ': kernel),
    out_sh ~ ConvTranspose3dShape '[batch, in_channels, depth, height, width] (in_channels ': Div out_channels groups ': kernel) stride padding dilation groups extra,
    Shape out_sh,
    CanBroadcast '[out_channels, 1, 1, 1] out_sh
  ) =>
  TransposedConvolution in_channels out_channels kernel groups t a ->
  arrow (t '[batch, in_channels, depth, height, width] a) (t out_sh a)
convTranspose3d parameters = arr $ \input -> T.convTranspose3d @stride @padding @dilation @groups @extra input (transposedWeights parameters) + broadcast (reshape @'[out_channels, 1, 1, 1] (transposedBias parameters))
