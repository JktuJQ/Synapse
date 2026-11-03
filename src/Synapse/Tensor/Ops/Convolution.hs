{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Grouped, dilated spatial convolutions and pooling in channels-first layout.
module Synapse.Tensor.Ops.Convolution (conv1d, conv2d, conv3d, convTranspose1d, convTranspose2d, convTranspose3d, maxPool1d, maxPool2d, maxPool3d, avgPool1d, avgPool2d, avgPool3d) where

import GHC.TypeNats (KnownNat)
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Convolution as Pullback
import Synapse.Tensor.Ops (Needs (Fractional, Numeric), TensorOps (customVJP))
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Primitives.Convolution as Raw
import Synapse.Tensor.Shapes.Convolution (Conv1dShape, Conv2dShape, Conv3dShape, ConvTranspose1dShape, ConvTranspose2dShape, ConvTranspose3dShape)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Window (Pool1dShape, Pool2dShape, Pool3dShape)
import Synapse.Tensor.Type (TensorElement)

-- | Apply 1D cross-correlation to NCL input using OIK weights, without reversing the kernel.
-- Type arguments specify stride, @[left, right]@ zero padding, dilation, and groups.
-- Weights have @input_channels / groups@ channels per output filter; input and output channels must be divisible by groups.
-- Effective kernel sizes are @dilation * (kernel - 1) + 1@ and output axes are @1 + (size + padding_before + padding_after - effective_kernel) div stride@.
-- Groups and all spatial kernel, stride and dilation values must be positive; no bias is applied.
conv1d :: forall stride padding dilation groups {t} input kernel a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape (Conv1dShape input kernel stride padding dilation groups)) => t input a -> t kernel a -> t (Conv1dShape input kernel stride padding dilation groups) a
conv1d =
  curry $
    customVJP
      Numeric
      (uncurry (Raw.convolution @stride @padding @dilation @groups))
      ( \(left, right) ->
          let output =
                (Raw.convolution @stride @padding @dilation @groups) left right
           in (output, ((left, right), output))
      )
      ( \((left, right), output) ->
          (Pullback.convolution @stride @padding @dilation @groups)
            left
            right
            output
      )

-- | Apply 2D cross-correlation to NCHW input using OIHW weights, without reversing the kernel.
-- Type arguments specify stride, @[top, bottom, left, right]@ zero padding, dilation, and groups.
-- Weights have @input_channels / groups@ channels per output filter; input and output channels must be divisible by groups.
-- Effective kernel sizes are @dilation * (kernel - 1) + 1@ and output axes are @1 + (size + padding_before + padding_after - effective_kernel) div stride@.
-- Groups and all spatial kernel, stride and dilation values must be positive; no bias is applied.
conv2d :: forall stride padding dilation groups {t} input kernel a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape (Conv2dShape input kernel stride padding dilation groups)) => t input a -> t kernel a -> t (Conv2dShape input kernel stride padding dilation groups) a
conv2d =
  curry $
    customVJP
      Numeric
      (uncurry (Raw.convolution @stride @padding @dilation @groups))
      ( \(left, right) ->
          let output =
                (Raw.convolution @stride @padding @dilation @groups) left right
           in (output, ((left, right), output))
      )
      ( \((left, right), output) ->
          (Pullback.convolution @stride @padding @dilation @groups)
            left
            right
            output
      )

-- | Apply 3D cross-correlation to NCDHW input using OIDHW weights, without reversing the kernel.
-- Type arguments specify stride, @[front, back, top, bottom, left, right]@ zero padding, dilation, and groups.
-- Weights have @input_channels / groups@ channels per output filter; input and output channels must be divisible by groups.
-- Effective kernel sizes are @dilation * (kernel - 1) + 1@ and output axes are @1 + (size + padding_before + padding_after - effective_kernel) div stride@.
-- Groups and all spatial kernel, stride and dilation values must be positive; no bias is applied.
conv3d :: forall stride padding dilation groups {t} input kernel a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape (Conv3dShape input kernel stride padding dilation groups)) => t input a -> t kernel a -> t (Conv3dShape input kernel stride padding dilation groups) a
conv3d =
  curry $
    customVJP
      Numeric
      (uncurry (Raw.convolution @stride @padding @dilation @groups))
      ( \(left, right) ->
          let output =
                (Raw.convolution @stride @padding @dilation @groups) left right
           in (output, ((left, right), output))
      )
      ( \((left, right), output) ->
          (Pullback.convolution @stride @padding @dilation @groups)
            left
            right
            output
      )

-- | Apply transposed 1D convolution to NCL input using IOK weights.
-- Type arguments specify stride, @[left, right]@ padding, dilation, groups, and @[extra_length]@ output padding.
-- Weights have @output_channels / groups@ channels per input channel.
-- Output axes are @(size - 1) * stride - padding_before - padding_after + dilation * (kernel - 1) + 1 + extra@.
-- Each extra size must be smaller than its stride; output padding selects a shape rather than appending zeros.
-- This is the transpose of the convolution mapping, not its inverse.
convTranspose1d :: forall stride padding dilation groups extra {t} input kernel a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape (ConvTranspose1dShape input kernel stride padding dilation groups extra)) => t input a -> t kernel a -> t (ConvTranspose1dShape input kernel stride padding dilation groups extra) a
convTranspose1d =
  curry $
    customVJP
      Numeric
      (uncurry (Raw.convTranspose @stride @padding @dilation @groups))
      ( \(left, right) ->
          let output =
                (Raw.convTranspose @stride @padding @dilation @groups) left right
           in (output, ((left, right), output))
      )
      ( \((left, right), output) ->
          (Pullback.convTranspose @stride @padding @dilation @groups)
            left
            right
            output
      )

-- | Apply transposed 2D convolution to NCHW input using IOHW weights.
-- Type arguments specify stride, @[top, bottom, left, right]@ padding, dilation, groups, and @[extra_height, extra_width]@ output padding.
-- Weights have @output_channels / groups@ channels per input channel.
-- Output axes are @(size - 1) * stride - padding_before - padding_after + dilation * (kernel - 1) + 1 + extra@.
-- Each extra size must be smaller than its stride; output padding selects a shape rather than appending zeros.
-- This is the transpose of the convolution mapping, not its inverse.
convTranspose2d :: forall stride padding dilation groups extra {t} input kernel a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape (ConvTranspose2dShape input kernel stride padding dilation groups extra)) => t input a -> t kernel a -> t (ConvTranspose2dShape input kernel stride padding dilation groups extra) a
convTranspose2d =
  curry $
    customVJP
      Numeric
      (uncurry (Raw.convTranspose @stride @padding @dilation @groups))
      ( \(left, right) ->
          let output =
                (Raw.convTranspose @stride @padding @dilation @groups) left right
           in (output, ((left, right), output))
      )
      ( \((left, right), output) ->
          (Pullback.convTranspose @stride @padding @dilation @groups)
            left
            right
            output
      )

-- | Apply transposed 3D convolution to NCDHW input using IODHW weights.
-- Type arguments specify stride, @[front, back, top, bottom, left, right]@ padding, dilation, groups, and @[extra_depth, extra_height, extra_width]@ output padding.
-- Weights have @output_channels / groups@ channels per input channel.
-- Output axes are @(size - 1) * stride - padding_before - padding_after + dilation * (kernel - 1) + 1 + extra@.
-- Each extra size must be smaller than its stride; output padding selects a shape rather than appending zeros.
-- This is the transpose of the convolution mapping, not its inverse.
convTranspose3d :: forall stride padding dilation groups extra {t} input kernel a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, KnownNat groups, Shape (ConvTranspose3dShape input kernel stride padding dilation groups extra)) => t input a -> t kernel a -> t (ConvTranspose3dShape input kernel stride padding dilation groups extra) a
convTranspose3d =
  curry $
    customVJP
      Numeric
      (uncurry (Raw.convTranspose @stride @padding @dilation @groups))
      ( \(left, right) ->
          let output =
                (Raw.convTranspose @stride @padding @dilation @groups) left right
           in (output, ((left, right), output))
      )
      ( \((left, right), output) ->
          (Pullback.convTranspose @stride @padding @dilation @groups)
            left
            right
            output
      )

-- | Maximum over 1D spatial windows in NCL layout.
-- Type arguments specify kernel, stride and @[left, right]@ padding, followed by dilation.
-- Padding is excluded from candidates; ties and NaN select the first row-major occurrence.
-- Windows containing only padding produce negative infinity for floating point or the minimum integral value and have zero gradient.
-- Padding on each side must not exceed half the undilated kernel size; output sizes use floor rounding.
maxPool1d :: forall kernel stride padding dilation {t} input a. (TensorOps t a, Num (TensorElement a), E.Ord (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape (Pool1dShape input kernel stride padding dilation)) => t input a -> t (Pool1dShape input kernel stride padding dilation) a
maxPool1d = customVJP Numeric (Raw.maxPool @kernel @stride @padding @dilation) (\primalInput -> let output = (Raw.maxPool @kernel @stride @padding @dilation) primalInput in (output, (primalInput, output))) (uncurry (Pullback.maxPool @kernel @stride @padding @dilation))

-- | Maximum over 2D spatial windows in NCHW layout.
-- Type arguments specify kernel, stride and @[top, bottom, left, right]@ padding, followed by dilation.
-- Padding is excluded from candidates; ties and NaN select the first row-major occurrence.
-- Windows containing only padding produce negative infinity for floating point or the minimum integral value and have zero gradient.
-- Padding on each side must not exceed half the undilated kernel size; output sizes use floor rounding.
maxPool2d :: forall kernel stride padding dilation {t} input a. (TensorOps t a, Num (TensorElement a), E.Ord (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape (Pool2dShape input kernel stride padding dilation)) => t input a -> t (Pool2dShape input kernel stride padding dilation) a
maxPool2d = customVJP Numeric (Raw.maxPool @kernel @stride @padding @dilation) (\primalInput -> let output = (Raw.maxPool @kernel @stride @padding @dilation) primalInput in (output, (primalInput, output))) (uncurry (Pullback.maxPool @kernel @stride @padding @dilation))

-- | Maximum over 3D spatial windows in NCDHW layout.
-- Type arguments specify kernel, stride and @[front, back, top, bottom, left, right]@ padding, followed by dilation.
-- Padding is excluded from candidates; ties and NaN select the first row-major occurrence.
-- Windows containing only padding produce negative infinity for floating point or the minimum integral value and have zero gradient.
-- Padding on each side must not exceed half the undilated kernel size; output sizes use floor rounding.
maxPool3d :: forall kernel stride padding dilation {t} input a. (TensorOps t a, Num (TensorElement a), E.Ord (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape (Pool3dShape input kernel stride padding dilation)) => t input a -> t (Pool3dShape input kernel stride padding dilation) a
maxPool3d = customVJP Numeric (Raw.maxPool @kernel @stride @padding @dilation) (\primalInput -> let output = (Raw.maxPool @kernel @stride @padding @dilation) primalInput in (output, (primalInput, output))) (uncurry (Pullback.maxPool @kernel @stride @padding @dilation))

-- | Average over 1D spatial windows in NCL layout.
-- Type arguments specify kernel, stride and @[left, right]@ padding.
-- The boolean argument includes padded zeros in the divisor when 'True'; 'False' counts only input elements.
-- Average pooling uses unit dilation.
-- Padding on each side must not exceed half the undilated kernel size; output sizes use floor rounding.
avgPool1d :: forall kernel stride padding {t} input a. (TensorOps t a, Fractional (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape (Pool1dShape input kernel stride padding '[1])) => Bool -> t input a -> t (Pool1dShape input kernel stride padding '[1]) a
avgPool1d includePadding = customVJP Fractional (Raw.avgPool @kernel @stride @padding includePadding) (\primalInput -> let output = Raw.avgPool @kernel @stride @padding includePadding primalInput in (output, (primalInput, output))) (uncurry (Pullback.avgPool @kernel @stride @padding includePadding))

-- | Average over 2D spatial windows in NCHW layout.
-- Type arguments specify kernel, stride and @[top, bottom, left, right]@ padding.
-- The boolean argument includes padded zeros in the divisor when 'True'; 'False' counts only input elements.
-- Average pooling uses unit dilation.
-- Padding on each side must not exceed half the undilated kernel size; output sizes use floor rounding.
avgPool2d :: forall kernel stride padding {t} input a. (TensorOps t a, Fractional (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape (Pool2dShape input kernel stride padding '[1, 1])) => Bool -> t input a -> t (Pool2dShape input kernel stride padding '[1, 1]) a
avgPool2d includePadding = customVJP Fractional (Raw.avgPool @kernel @stride @padding includePadding) (\primalInput -> let output = Raw.avgPool @kernel @stride @padding includePadding primalInput in (output, (primalInput, output))) (uncurry (Pullback.avgPool @kernel @stride @padding includePadding))

-- | Average over 3D spatial windows in NCDHW layout.
-- Type arguments specify kernel, stride and @[front, back, top, bottom, left, right]@ padding.
-- The boolean argument includes padded zeros in the divisor when 'True'; 'False' counts only input elements.
-- Average pooling uses unit dilation.
-- Padding on each side must not exceed half the undilated kernel size; output sizes use floor rounding.
avgPool3d :: forall kernel stride padding {t} input a. (TensorOps t a, Fractional (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape (Pool3dShape input kernel stride padding '[1, 1, 1])) => Bool -> t input a -> t (Pool3dShape input kernel stride padding '[1, 1, 1]) a
avgPool3d includePadding = customVJP Fractional (Raw.avgPool @kernel @stride @padding includePadding) (\primalInput -> let output = Raw.avgPool @kernel @stride @padding includePadding primalInput in (output, (primalInput, output))) (uncurry (Pullback.avgPool @kernel @stride @padding includePadding))
