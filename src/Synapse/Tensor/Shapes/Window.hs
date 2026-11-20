{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Output shapes for spatial windows, patches, pooling, and convolutions.
module Synapse.Tensor.Shapes.Window
    ( -- * Spatial output dimensions
      WindowSpatialShape

      -- * Extracted patches
    , Patches1dShape
    , Patches2dShape
    , Patches3dShape

      -- * Pooling
    , Pool1dShape
    , Pool2dShape
    , Pool3dShape

      -- * Convolution
    , Conv1dShape
    , Conv2dShape
    , Conv3dShape

      -- * Transposed convolution
    , ConvTranspose1dShape
    , ConvTranspose2dShape
    , ConvTranspose3dShape
    ) where

import GHC.TypeLits (CmpNat, Div, ErrorMessage (Text), Mod, Nat, TypeError, type (*), type (+), type (-))

import Synapse.Tensor.Shapes.Dimensions (ShapeSize)

-- Spatial window dimensions

-- | Require the effective kernel to fit the padded input before computing a floor-rounded output size.
type family
    PaddedExtent
        (order :: Ordering)
        (size :: Nat)
        (kernel :: Nat)
        (before :: Nat)
        (after :: Nat)
        (stride :: Nat)
        :: Nat
    where
    PaddedExtent 'GT size kernel before after stride = TypeError ('Text "Window kernel exceeds the padded input")
    PaddedExtent order size kernel before after stride = 1 + Div (size + before + after - kernel) stride

{- | Output size of one dilated sliding-window axis.
Reject zero kernels, strides, or dilations before checking that the effective kernel fits the padded input.
-}
type family
    WindowExtent
        (size :: Nat)
        (kernel :: Nat)
        (before :: Nat)
        (after :: Nat)
        (stride :: Nat)
        (dilation :: Nat)
        :: Nat
    where
    WindowExtent size 0 before after stride dilation =
        TypeError ('Text "Window kernel dimensions must be positive")
    WindowExtent size kernel before after 0 dilation = TypeError ('Text "Window strides must be positive")
    WindowExtent size kernel before after stride 0 = TypeError ('Text "Window dilation must be positive")
    WindowExtent size kernel before after stride dilation =
        PaddedExtent
            (CmpNat (dilation * (kernel - 1) + 1) (size + before + after))
            size
            (dilation * (kernel - 1) + 1)
            before
            after
            stride

{- | Spatial output dimensions for a dilated sliding window.

Arguments are the spatial input shape, kernel sizes, strides, padding pairs, and dilations, in that order.
Kernel, stride, and dilation lists must each have one entry per spatial axis;
padding must have two entries per axis, before then after.
Each output dimension uses the floor-rounded window formula described in this module.

@
WindowSpatialShape '[5, 5] '[3, 3] '[1, 1] '[0, 0, 0, 0] '[1, 1] = '[3, 3]
@
-}
type family
    WindowSpatialShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    WindowSpatialShape '[] '[] '[] '[] '[] = '[]
    WindowSpatialShape
        (size ': sizes)
        (kernel ': kernels)
        (stride ': strides)
        (before ': after ': padding)
        (dilation ': dilations) =
        WindowExtent size kernel before after stride dilation
            ': WindowSpatialShape sizes kernels strides padding dilations
    WindowSpatialShape input kernel stride padding dilation =
        TypeError ('Text "Window kernel, stride, padding and dilation ranks must match the spatial rank")

-- Patch extraction

-- | Collect channels and kernel coordinates into patch rows, and spatial output coordinates into columns.
type family
    PatchesShape
        (batch :: Nat)
        (channels :: Nat)
        (spatial :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    PatchesShape batch channels spatial kernel stride padding dilation =
        '[batch, channels * ShapeSize kernel, ShapeSize (WindowSpatialShape spatial kernel stride padding dilation)]

{- | Shape of 1D patches extracted from @[batch, channels, length]@ input.

Arguments are input, kernel, stride, padding, and dilation shapes.
Spatial parameters have shapes @'[k]@, @'[s]@, @'[left, right]@, and @'[d]@.
The result is @'[batch, channels * k, positions]@, with positions computed by 'WindowSpatialShape'.
-}
type family
    Patches1dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    Patches1dShape '[batch, channels, size] '[k] '[s] '[left, right] '[d] =
        PatchesShape batch channels '[size] '[k] '[s] '[left, right] '[d]
    Patches1dShape input kernel stride padding dilation =
        TypeError ('Text "Patches1d requires NCL input and matching kernel, stride, padding and dilation ranks")

{- | Shape of 2D patches extracted from @[batch, channels, height, width]@ input.

Arguments are input, kernel, stride, padding, and dilation shapes.
Spatial parameters are @'[kh, kw]@, @'[sy, sx]@, @'[top, bottom, left, right]@, and @'[dy, dx]@.
The result is @'[batch, channels * kh * kw, positions]@;
@positions@ is the product of spatial output dimensions from 'WindowSpatialShape'.
-}
type family
    Patches2dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    Patches2dShape '[batch, channels, height, width] '[kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx] =
        PatchesShape batch channels '[height, width] '[kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx]
    Patches2dShape input kernel stride padding dilation =
        TypeError ('Text "Patches2d requires NCHW input and matching kernel, stride, padding and dilation ranks")

{- | Shape of 3D patches extracted from @[batch, channels, depth, height, width]@ input.

Arguments are input, kernel, stride, padding, and dilation shapes.
Kernel, stride, and dilation lists follow depth, height, width order;
padding is @'[front, back, top, bottom, left, right]@.
The result is @'[batch, channels * kd * kh * kw, positions]@;
@positions@ is the product of spatial output dimensions from 'WindowSpatialShape'.
-}
type family
    Patches3dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    Patches3dShape
        '[batch, channels, depth, height, width]
        '[kd, kh, kw]
        '[sz, sy, sx]
        '[front, back, top, bottom, left, right]
        '[dz, dy, dx] =
        PatchesShape
            batch
            channels
            '[depth, height, width]
            '[kd, kh, kw]
            '[sz, sy, sx]
            '[front, back, top, bottom, left, right]
            '[dz, dy, dx]
    Patches3dShape input kernel stride padding dilation =
        TypeError ('Text "Patches3d requires NCDHW input and matching kernel, stride, padding and dilation ranks")

-- Pooling

-- | Reject padding on either side that exceeds half the undilated kernel size.
type family
    PoolPadding
        (before :: Ordering)
        (after :: Ordering)
        (result :: Nat)
        :: Nat
    where
    PoolPadding 'GT after result = TypeError ('Text "Pooling padding must not exceed half the kernel size")
    PoolPadding before 'GT result = TypeError ('Text "Pooling padding must not exceed half the kernel size")
    PoolPadding before after result = result

-- | Require a positive spatial input size and valid pooling padding before computing the window extent.
type family
    PoolExtent
        (size :: Nat)
        (kernel :: Nat)
        (before :: Nat)
        (after :: Nat)
        (stride :: Nat)
        (dilation :: Nat)
        :: Nat
    where
    PoolExtent 0 kernel before after stride dilation =
        TypeError ('Text "Pooling spatial input dimensions must be positive")
    PoolExtent size kernel before after stride dilation =
        PoolPadding
            (CmpNat before (Div kernel 2))
            (CmpNat after (Div kernel 2))
            (WindowExtent size kernel before after stride dilation)

-- | Validate pooling parameters and compute output dimensions independently along each spatial axis.
type family
    PoolSpatialShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    PoolSpatialShape '[] '[] '[] '[] '[] = '[]
    PoolSpatialShape
        (size ': sizes)
        (kernel ': kernels)
        (stride ': strides)
        (before ': after ': padding)
        (dilation ': dilations) =
        PoolExtent size kernel before after stride dilation
            ': PoolSpatialShape sizes kernels strides padding dilations
    PoolSpatialShape input kernel stride padding dilation =
        TypeError ('Text "Pooling spatial parameter ranks must match")

{- | Shape of 1D pooling over @[batch, channels, length]@ input.

Arguments are input, kernel, stride, padding, and dilation shapes.
Spatial parameters are @'[k]@, @'[s]@, @'[left, right]@, and @'[d]@.
The result preserves batch and channels and uses a floor-rounded spatial output size.
The input length must be positive; each padding value must not exceed @floor (k / 2)@.
-}
type family
    Pool1dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    Pool1dShape '[batch, channels, size] '[k] '[s] '[left, right] '[d] =
        batch ': channels ': PoolSpatialShape '[size] '[k] '[s] '[left, right] '[d]
    Pool1dShape input kernel stride padding dilation =
        TypeError ('Text "Pool1d requires NCL input and matching kernel, stride, padding and dilation ranks")

{- | Shape of 2D pooling over @[batch, channels, height, width]@ input.

Arguments are input, kernel, stride, padding, and dilation shapes.
Spatial parameters are @'[kh, kw]@, @'[sy, sx]@, @'[top, bottom, left, right]@, and @'[dy, dx]@.
The result preserves batch and channels and uses floor-rounded spatial output sizes.
Spatial input sizes must be positive; padding on either side must not exceed half its kernel size.
-}
type family
    Pool2dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    Pool2dShape '[batch, channels, height, width] '[kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx] =
        batch
            ': channels
            ': PoolSpatialShape '[height, width] '[kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx]
    Pool2dShape input kernel stride padding dilation =
        TypeError ('Text "Pool2d requires NCHW input and matching kernel, stride, padding and dilation ranks")

{- | Shape of 3D pooling over @[batch, channels, depth, height, width]@ input.

Arguments are input, kernel, stride, padding, and dilation shapes.
Kernel, stride, and dilation lists follow depth, height, width order;
padding is @'[front, back, top, bottom, left, right]@.
The result preserves batch and channels and uses floor-rounded spatial output sizes.
Spatial input sizes must be positive; padding on either side must not exceed half its kernel size.
-}
type family
    Pool3dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        :: [Nat]
    where
    Pool3dShape
        '[batch, channels, depth, height, width]
        '[kd, kh, kw]
        '[sz, sy, sx]
        '[front, back, top, bottom, left, right]
        '[dz, dy, dx] =
        batch
            ': channels
            ': PoolSpatialShape
                '[depth, height, width]
                '[kd, kh, kw]
                '[sz, sy, sx]
                '[front, back, top, bottom, left, right]
                '[dz, dy, dx]
    Pool3dShape input kernel stride padding dilation =
        TypeError ('Text "Pool3d requires NCDHW input and matching kernel, stride, padding and dilation ranks")

-- Convolution

-- | Require divisible channel counts and matching input channels per filter within each group.
type family
    GroupChannels (cinRemainder :: Nat) (coutRemainder :: Nat) (expected :: Nat) (actual :: Nat) (out :: [Nat])
        :: [Nat]
    where
    GroupChannels 0 0 channels channels out = out
    GroupChannels 0 0 expected actual out =
        TypeError ('Text "Convolution input and weight channels must match within each group")
    GroupChannels cinRemainder coutRemainder expected actual out =
        TypeError ('Text "Convolution input and output channels must be divisible by groups")

-- | Require a positive group count before dividing input channels into groups.
type family ConvGroups (groups :: Nat) (cin :: Nat) (cout :: Nat) (win :: Nat) (out :: [Nat]) :: [Nat] where
    ConvGroups 0 cin cout win out = TypeError ('Text "Convolution groups must be positive")
    ConvGroups groups cin cout win out = GroupChannels (Mod cin groups) (Mod cout groups) (Div cin groups) win out

-- | Compute grouped convolution channels and spatial window dimensions from channel-first input and weights.
type family
    ConvShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (groups :: Nat)
        :: [Nat]
    where
    ConvShape (batch ': cin ': spatial) (cout ': win ': kernel) stride padding dilation groups =
        ConvGroups groups cin cout win (batch ': cout ': WindowSpatialShape spatial kernel stride padding dilation)

{- | Output shape of a grouped, dilated 1D convolution.

Arguments are input shape, weight shape, strides, padding, dilations, and group count.
Input layout is @[batch, input_channels, length]@;
weights use @[output_channels, input_channels / groups, k]@.
The output retains the batch size and replaces input channels with output channels.

Stride and dilation lists follow length order; padding is @'[left, right]@.
Spatial output sizes use the floor-rounded formula in "Synapse.Tensor.Shapes.Window".
Groups must be positive and divide both channel counts; weight channels must match one input group.
Kernels, strides, and dilations must be positive, and the effective kernel must fit the padded input.
-}
type family
    Conv1dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (groups :: Nat)
        :: [Nat]
    where
    Conv1dShape '[batch, cin, size] '[cout, win, k] '[s] '[left, right] '[d] groups =
        ConvShape '[batch, cin, size] '[cout, win, k] '[s] '[left, right] '[d] groups
    Conv1dShape input kernel stride padding dilation groups =
        TypeError ('Text "Conv1d requires NCL inputs, OIK weights and matching spatial parameter ranks")

{- | Output shape of a grouped, dilated 2D convolution.

Arguments are input shape, weight shape, strides, padding, dilations, and group count.
Input layout is @[batch, input_channels, height, width]@;
weights use @[output_channels, input_channels / groups, kh, kw]@.
The output retains the batch size and replaces input channels with output channels.

Stride and dilation lists follow height, width order; padding is @'[top, bottom, left, right]@.
Spatial output sizes use the floor-rounded formula in "Synapse.Tensor.Shapes.Window".
Groups must be positive and divide both channel counts; weight channels must match one input group.
Kernels, strides, and dilations must be positive, and the effective kernel must fit the padded input.
-}
type family
    Conv2dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (groups :: Nat)
        :: [Nat]
    where
    Conv2dShape
        '[batch, cin, height, width]
        '[cout, win, kh, kw]
        '[sy, sx]
        '[top, bottom, left, right]
        '[dy, dx]
        groups =
        ConvShape
            '[batch, cin, height, width]
            '[cout, win, kh, kw]
            '[sy, sx]
            '[top, bottom, left, right]
            '[dy, dx]
            groups
    Conv2dShape input kernel stride padding dilation groups =
        TypeError ('Text "Conv2d requires NCHW inputs, OIHW weights and matching spatial parameter ranks")

{- | Output shape of a grouped, dilated 3D convolution.

Arguments are input shape, weight shape, strides, padding, dilations, and group count.
Input layout is @[batch, input_channels, depth, height, width]@;
weights use @[output_channels, input_channels / groups, kd, kh, kw]@.
The output retains the batch size and replaces input channels with output channels.

Stride and dilation lists follow depth, height, width order;
padding is @'[front, back, top, bottom, left, right]@.
Spatial output sizes use the floor-rounded formula in "Synapse.Tensor.Shapes.Window".
Groups must be positive and divide both channel counts; weight channels must match one input group.
Kernels, strides, and dilations must be positive, and the effective kernel must fit the padded input.
-}
type family
    Conv3dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (groups :: Nat)
        :: [Nat]
    where
    Conv3dShape
        '[batch, cin, depth, height, width]
        '[cout, win, kd, kh, kw]
        '[sz, sy, sx]
        '[front, back, top, bottom, left, right]
        '[dz, dy, dx]
        groups =
        ConvShape
            '[batch, cin, depth, height, width]
            '[cout, win, kd, kh, kw]
            '[sz, sy, sx]
            '[front, back, top, bottom, left, right]
            '[dz, dy, dx]
            groups
    Conv3dShape input kernel stride padding dilation groups =
        TypeError ('Text "Conv3d requires NCDHW inputs, OIDHW weights and matching spatial parameter ranks")

-- Transposed convolution: spatial dimensions

-- | Subtract total padding only when the resulting transposed-convolution extent is positive.
type family TransposeSize (order :: Ordering) (size :: Nat) (padding :: Nat) :: Nat where
    TransposeSize 'GT size padding = size - padding
    TransposeSize order size padding =
        TypeError ('Text "Transposed convolution output dimensions must be positive")

-- | Require output padding smaller than the stride before subtracting spatial padding.
type family
    TransposePadding
        (order :: Ordering)
        (size :: Nat)
        (kernel :: Nat)
        (before :: Nat)
        (after :: Nat)
        (stride :: Nat)
        (extra :: Nat)
        :: Nat
    where
    TransposePadding 'LT size kernel before after stride extra =
        TransposeSize
            (CmpNat ((size - 1) * stride + kernel + extra) (before + after))
            ((size - 1) * stride + kernel + extra)
            (before + after)
    TransposePadding order size kernel before after stride extra =
        TypeError ('Text "Transposed convolution output padding must be smaller than stride")

-- | Validate one transposed-convolution axis and compute its output size using the effective kernel.
type family
    TransposeExtent
        (size :: Nat)
        (kernel :: Nat)
        (before :: Nat)
        (after :: Nat)
        (stride :: Nat)
        (dilation :: Nat)
        (extra :: Nat)
        :: Nat
    where
    TransposeExtent 0 kernel before after stride dilation extra =
        TypeError ('Text "Transposed convolution spatial input dimensions must be positive")
    TransposeExtent size 0 before after stride dilation extra =
        TypeError ('Text "Window kernel dimensions must be positive")
    TransposeExtent size kernel before after 0 dilation extra =
        TypeError ('Text "Window strides must be positive")
    TransposeExtent size kernel before after stride 0 extra = TypeError ('Text "Window dilation must be positive")
    TransposeExtent size kernel before after stride dilation extra =
        TransposePadding (CmpNat extra stride) size (dilation * (kernel - 1) + 1) before after stride extra

-- | Compute transposed-convolution output extents with one output-padding value per spatial axis.
type family
    TransposeSpatialShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (extra :: [Nat])
        :: [Nat]
    where
    TransposeSpatialShape '[] '[] '[] '[] '[] '[] = '[]
    TransposeSpatialShape
        (size ': sizes)
        (kernel ': kernels)
        (stride ': strides)
        (before ': after ': padding)
        (dilation ': dilations)
        (extra ': extras) =
        TransposeExtent size kernel before after stride dilation extra
            ': TransposeSpatialShape sizes kernels strides padding dilations extras
    TransposeSpatialShape input kernel stride padding dilation extra =
        TypeError ('Text "Transposed convolution spatial parameter ranks must match")

-- Transposed convolution: channels

-- | Require input channels divisible by groups and equal to the first weight dimension.
type family TransposeChannels (remainder :: Nat) (cin :: Nat) (win :: Nat) (out :: [Nat]) :: [Nat] where
    TransposeChannels 0 channels channels out = out
    TransposeChannels 0 cin win out = TypeError ('Text "Convolution input and weight channels must match")
    TransposeChannels remainder cin win out =
        TypeError ('Text "Convolution input channels must be divisible by groups")

-- | Require positive groups before validating transposed-convolution input channels.
type family TransposeGroups (groups :: Nat) (cin :: Nat) (win :: Nat) (out :: [Nat]) :: [Nat] where
    TransposeGroups 0 cin win out = TypeError ('Text "Convolution groups must be positive")
    TransposeGroups groups cin win out = TransposeChannels (Mod cin groups) cin win out

-- Transposed convolution: output shapes

{- | Compute transposed-convolution channels and spatial extents.
Weights specify input channels and output channels per group.
-}
type family
    ConvTransposeShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (groups :: Nat)
        (extra :: [Nat])
        :: [Nat]
    where
    ConvTransposeShape (batch ': cin ': spatial) (win ': cout ': kernel) stride padding dilation groups extra =
        TransposeGroups
            groups
            cin
            win
            (batch ': (cout * groups) ': TransposeSpatialShape spatial kernel stride padding dilation extra)

{- | Output shape of a grouped, dilated 1D transposed convolution.

Arguments are input shape, weight shape, strides, padding, dilations, groups, and output padding.
Input layout is @[batch, input_channels, length]@;
weights use @[input_channels, output_channels / groups, k]@.
The output channel count is the second weight dimension multiplied by groups.

Spatial lists follow length order; padding is @'[left, right]@.
For each axis, the output size is
@(input - 1) * stride - before - after + dilation * (kernel - 1) + 1 + extra@.
Output padding @extra@ has one entry per spatial axis and must be smaller than its stride.
Groups must be positive and divide input channels; the first weight dimension must match input channels.
Kernels, strides, dilations, and spatial input and output sizes must be positive.
-}
type family
    ConvTranspose1dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (groups :: Nat)
        (extra :: [Nat])
        :: [Nat]
    where
    ConvTranspose1dShape '[batch, cin, size] '[win, cout, k] '[s] '[left, right] '[d] groups '[extra] =
        ConvTransposeShape '[batch, cin, size] '[win, cout, k] '[s] '[left, right] '[d] groups '[extra]
    ConvTranspose1dShape input kernel stride padding dilation groups extra =
        TypeError ('Text "ConvTranspose1d requires NCL inputs, IOK weights and matching spatial parameter ranks")

{- | Output shape of a grouped, dilated 2D transposed convolution.

Arguments are input shape, weight shape, strides, padding, dilations, groups, and output padding.
Input layout is @[batch, input_channels, height, width]@;
weights use @[input_channels, output_channels / groups, kh, kw]@.
The output channel count is the second weight dimension multiplied by groups.

Spatial lists follow height, width order; padding is @'[top, bottom, left, right]@.
For each axis, the output size is
@(input - 1) * stride - before - after + dilation * (kernel - 1) + 1 + extra@.
Output padding @extra@ has one entry per spatial axis and must be smaller than its stride.
Groups must be positive and divide input channels; the first weight dimension must match input channels.
Kernels, strides, dilations, and spatial input and output sizes must be positive.
-}
type family
    ConvTranspose2dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (groups :: Nat)
        (extra :: [Nat])
        :: [Nat]
    where
    ConvTranspose2dShape
        '[batch, cin, height, width]
        '[win, cout, kh, kw]
        '[sy, sx]
        '[top, bottom, left, right]
        '[dy, dx]
        groups
        '[ey, ex] =
        ConvTransposeShape
            '[batch, cin, height, width]
            '[win, cout, kh, kw]
            '[sy, sx]
            '[top, bottom, left, right]
            '[dy, dx]
            groups
            '[ey, ex]
    ConvTranspose2dShape input kernel stride padding dilation groups extra =
        TypeError ('Text "ConvTranspose2d requires NCHW inputs, IOHW weights and matching spatial parameter ranks")

{- | Output shape of a grouped, dilated 3D transposed convolution.

Arguments are input shape, weight shape, strides, padding, dilations, groups, and output padding.
Input layout is @[batch, input_channels, depth, height, width]@;
weights use @[input_channels, output_channels / groups, kd, kh, kw]@.
The output channel count is the second weight dimension multiplied by groups.

Spatial lists follow depth, height, width order; padding is @'[front, back, top, bottom, left, right]@.
For each axis, the output size is
@(input - 1) * stride - before - after + dilation * (kernel - 1) + 1 + extra@.
Output padding @extra@ has one entry per spatial axis and must be smaller than its stride.
Groups must be positive and divide input channels; the first weight dimension must match input channels.
Kernels, strides, dilations, and spatial input and output sizes must be positive.
-}
type family
    ConvTranspose3dShape
        (input :: [Nat])
        (kernel :: [Nat])
        (stride :: [Nat])
        (padding :: [Nat])
        (dilation :: [Nat])
        (groups :: Nat)
        (extra :: [Nat])
        :: [Nat]
    where
    ConvTranspose3dShape
        '[batch, cin, depth, height, width]
        '[win, cout, kd, kh, kw]
        '[sz, sy, sx]
        '[front, back, top, bottom, left, right]
        '[dz, dy, dx]
        groups
        '[ez, ey, ex] =
        ConvTransposeShape
            '[batch, cin, depth, height, width]
            '[win, cout, kd, kh, kw]
            '[sz, sy, sx]
            '[front, back, top, bottom, left, right]
            '[dz, dy, dx]
            groups
            '[ez, ey, ex]
    ConvTranspose3dShape input kernel stride padding dilation groups extra =
        TypeError ('Text "ConvTranspose3d requires NCDHW inputs, IODHW weights and matching spatial parameter ranks")
