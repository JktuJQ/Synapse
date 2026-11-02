{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Output shapes and channel checks for grouped, dilated spatial convolutions.
module Synapse.Tensor.Shapes.Convolution (Conv1dShape, Conv2dShape, Conv3dShape, ConvTranspose1dShape, ConvTranspose2dShape, ConvTranspose3dShape) where

import GHC.TypeLits (CmpNat, Div, ErrorMessage (Text), Mod, Nat, TypeError, type (*), type (+), type (-))
import Synapse.Tensor.Shapes.Window (WindowSpatialShape)

-- | Compare the weight's input-channel count with the channels in one group.
type family GroupChannels (cinRemainder :: Nat) (coutRemainder :: Nat) (expected :: Nat) (actual :: Nat) (out :: [Nat]) :: [Nat] where
  GroupChannels 0 0 channels channels out = out
  GroupChannels 0 0 expected actual out = TypeError ('Text "Convolution input and weight channels must match within each group")
  GroupChannels cinRemainder coutRemainder expected actual out = TypeError ('Text "Convolution input and output channels must be divisible by groups")

-- | Check groups before dividing the input channel count.
type family ConvGroups (groups :: Nat) (cin :: Nat) (cout :: Nat) (win :: Nat) (out :: [Nat]) :: [Nat] where
  ConvGroups 0 cin cout win out = TypeError ('Text "Convolution groups must be positive")
  ConvGroups groups cin cout win out = GroupChannels (Mod cin groups) (Mod cout groups) (Div cin groups) win out

type family ConvShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (groups :: Nat) :: [Nat] where
  ConvShape (batch ': cin ': spatial) (cout ': win ': kernel) stride padding dilation groups = ConvGroups groups cin cout win (batch ': cout ': WindowSpatialShape spatial kernel stride padding dilation)

-- | NCL convolution with OIK weights containing @input_channels / groups@ channels per filter.
-- Effective kernel sizes are @dilation * (kernel - 1) + 1@; output axes use floor rounding.
-- Groups, strides, dilation and kernel sizes must be positive; both channel counts must be divisible by groups.
type family Conv1dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (groups :: Nat) :: [Nat] where
  Conv1dShape '[batch, cin, size] '[cout, win, k] '[s] '[left, right] '[d] groups = ConvShape '[batch, cin, size] '[cout, win, k] '[s] '[left, right] '[d] groups
  Conv1dShape input kernel stride padding dilation groups = TypeError ('Text "Conv1d requires NCL inputs, OIK weights and matching spatial parameter ranks")

-- | NCHW convolution with OIHW weights containing @input_channels / groups@ channels per filter.
-- Effective kernel sizes are @dilation * (kernel - 1) + 1@; output axes use floor rounding.
-- Groups, strides, dilation and kernel sizes must be positive; both channel counts must be divisible by groups.
type family Conv2dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (groups :: Nat) :: [Nat] where
  Conv2dShape '[batch, cin, height, width] '[cout, win, kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx] groups = ConvShape '[batch, cin, height, width] '[cout, win, kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx] groups
  Conv2dShape input kernel stride padding dilation groups = TypeError ('Text "Conv2d requires NCHW inputs, OIHW weights and matching spatial parameter ranks")

-- | NCDHW convolution with OIDHW weights containing @input_channels / groups@ channels per filter.
-- Effective kernel sizes are @dilation * (kernel - 1) + 1@; output axes use floor rounding.
-- Groups, strides, dilation and kernel sizes must be positive; both channel counts must be divisible by groups.
type family Conv3dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (groups :: Nat) :: [Nat] where
  Conv3dShape '[batch, cin, depth, height, width] '[cout, win, kd, kh, kw] '[sz, sy, sx] '[front, back, top, bottom, left, right] '[dz, dy, dx] groups = ConvShape '[batch, cin, depth, height, width] '[cout, win, kd, kh, kw] '[sz, sy, sx] '[front, back, top, bottom, left, right] '[dz, dy, dx] groups
  Conv3dShape input kernel stride padding dilation groups = TypeError ('Text "Conv3d requires NCDHW inputs, OIDHW weights and matching spatial parameter ranks")

type family TransposeSize (order :: Ordering) (size :: Nat) (padding :: Nat) :: Nat where
  TransposeSize 'GT size padding = size - padding
  TransposeSize order size padding = TypeError ('Text "Transposed convolution output dimensions must be positive")

type family TransposePadding (order :: Ordering) (size :: Nat) (kernel :: Nat) (before :: Nat) (after :: Nat) (stride :: Nat) (extra :: Nat) :: Nat where
  TransposePadding 'LT size kernel before after stride extra = TransposeSize (CmpNat ((size - 1) * stride + kernel + extra) (before + after)) ((size - 1) * stride + kernel + extra) (before + after)
  TransposePadding order size kernel before after stride extra = TypeError ('Text "Transposed convolution output padding must be smaller than stride")

type family TransposeExtent (size :: Nat) (kernel :: Nat) (before :: Nat) (after :: Nat) (stride :: Nat) (dilation :: Nat) (extra :: Nat) :: Nat where
  TransposeExtent 0 kernel before after stride dilation extra = TypeError ('Text "Transposed convolution spatial input dimensions must be positive")
  TransposeExtent size 0 before after stride dilation extra = TypeError ('Text "Window kernel dimensions must be positive")
  TransposeExtent size kernel before after 0 dilation extra = TypeError ('Text "Window strides must be positive")
  TransposeExtent size kernel before after stride 0 extra = TypeError ('Text "Window dilation must be positive")
  TransposeExtent size kernel before after stride dilation extra = TransposePadding (CmpNat extra stride) size (dilation * (kernel - 1) + 1) before after stride extra

type family TransposeSpatialShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (extra :: [Nat]) :: [Nat] where
  TransposeSpatialShape '[] '[] '[] '[] '[] '[] = '[]
  TransposeSpatialShape (size ': sizes) (kernel ': kernels) (stride ': strides) (before ': after ': padding) (dilation ': dilations) (extra ': extras) = TransposeExtent size kernel before after stride dilation extra ': TransposeSpatialShape sizes kernels strides padding dilations extras
  TransposeSpatialShape input kernel stride padding dilation extra = TypeError ('Text "Transposed convolution spatial parameter ranks must match")

type family TransposeChannels (remainder :: Nat) (cin :: Nat) (win :: Nat) (out :: [Nat]) :: [Nat] where
  TransposeChannels 0 channels channels out = out
  TransposeChannels 0 cin win out = TypeError ('Text "Convolution input and weight channels must match")
  TransposeChannels remainder cin win out = TypeError ('Text "Convolution input channels must be divisible by groups")

type family TransposeGroups (groups :: Nat) (cin :: Nat) (win :: Nat) (out :: [Nat]) :: [Nat] where
  TransposeGroups 0 cin win out = TypeError ('Text "Convolution groups must be positive")
  TransposeGroups groups cin win out = TransposeChannels (Mod cin groups) cin win out

type family ConvTransposeShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (groups :: Nat) (extra :: [Nat]) :: [Nat] where
  ConvTransposeShape (batch ': cin ': spatial) (win ': cout ': kernel) stride padding dilation groups extra = TransposeGroups groups cin win (batch ': (cout * groups) ': TransposeSpatialShape spatial kernel stride padding dilation extra)

-- | NCL transposed convolution with IOK weights containing @output_channels / groups@ channels per filter.
-- Output axes are @(size - 1) * stride - padding_before - padding_after + dilation * (kernel - 1) + 1 + extra@.
-- Each extra size must be smaller than its stride, and spatial input and output sizes must be positive.
type family ConvTranspose1dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (groups :: Nat) (extra :: [Nat]) :: [Nat] where
  ConvTranspose1dShape '[batch, cin, size] '[win, cout, k] '[s] '[left, right] '[d] groups '[extra] = ConvTransposeShape '[batch, cin, size] '[win, cout, k] '[s] '[left, right] '[d] groups '[extra]
  ConvTranspose1dShape input kernel stride padding dilation groups extra = TypeError ('Text "ConvTranspose1d requires NCL inputs, IOK weights and matching spatial parameter ranks")

-- | NCHW transposed convolution with IOHW weights containing @output_channels / groups@ channels per filter.
-- Output axes are @(size - 1) * stride - padding_before - padding_after + dilation * (kernel - 1) + 1 + extra@.
-- Each extra size must be smaller than its stride, and spatial input and output sizes must be positive.
type family ConvTranspose2dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (groups :: Nat) (extra :: [Nat]) :: [Nat] where
  ConvTranspose2dShape '[batch, cin, height, width] '[win, cout, kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx] groups '[ey, ex] = ConvTransposeShape '[batch, cin, height, width] '[win, cout, kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx] groups '[ey, ex]
  ConvTranspose2dShape input kernel stride padding dilation groups extra = TypeError ('Text "ConvTranspose2d requires NCHW inputs, IOHW weights and matching spatial parameter ranks")

-- | NCDHW transposed convolution with IODHW weights containing @output_channels / groups@ channels per filter.
-- Output axes are @(size - 1) * stride - padding_before - padding_after + dilation * (kernel - 1) + 1 + extra@.
-- Each extra size must be smaller than its stride, and spatial input and output sizes must be positive.
type family ConvTranspose3dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) (groups :: Nat) (extra :: [Nat]) :: [Nat] where
  ConvTranspose3dShape '[batch, cin, depth, height, width] '[win, cout, kd, kh, kw] '[sz, sy, sx] '[front, back, top, bottom, left, right] '[dz, dy, dx] groups '[ez, ey, ex] = ConvTransposeShape '[batch, cin, depth, height, width] '[win, cout, kd, kh, kw] '[sz, sy, sx] '[front, back, top, bottom, left, right] '[dz, dy, dx] groups '[ez, ey, ex]
  ConvTranspose3dShape input kernel stride padding dilation groups extra = TypeError ('Text "ConvTranspose3d requires NCDHW inputs, IODHW weights and matching spatial parameter ranks")
