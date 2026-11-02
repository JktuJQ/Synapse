{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Static shapes for spatial windows and pooling.
module Synapse.Tensor.Shapes.Window (WindowSpatialShape, Patches1dShape, Patches2dShape, Patches3dShape, Pool1dShape, Pool2dShape, Pool3dShape) where

import GHC.TypeLits (CmpNat, Div, ErrorMessage (Text), Nat, TypeError, type (*), type (+), type (-))
import Synapse.Tensor.Shapes.Dimensions (ShapeSize)

-- | Subtract only after checking that the kernel fits the padded input.
type family PaddedExtent (order :: Ordering) (size :: Nat) (kernel :: Nat) (before :: Nat) (after :: Nat) (stride :: Nat) :: Nat where
  PaddedExtent 'GT size kernel before after stride = TypeError ('Text "Window kernel exceeds the padded input")
  PaddedExtent order size kernel before after stride = 1 + Div (size + before + after - kernel) stride

-- | Size of a dilated sliding-window axis.
type family WindowExtent (size :: Nat) (kernel :: Nat) (before :: Nat) (after :: Nat) (stride :: Nat) (dilation :: Nat) :: Nat where
  WindowExtent size 0 before after stride dilation = TypeError ('Text "Window kernel dimensions must be positive")
  WindowExtent size kernel before after 0 dilation = TypeError ('Text "Window strides must be positive")
  WindowExtent size kernel before after stride 0 = TypeError ('Text "Window dilation must be positive")
  WindowExtent size kernel before after stride dilation = PaddedExtent (CmpNat (dilation * (kernel - 1) + 1) (size + before + after)) size (dilation * (kernel - 1) + 1) before after stride

-- | Spatial output sizes with one dilation and stride per axis and before/after padding pairs.
type family WindowSpatialShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  WindowSpatialShape '[] '[] '[] '[] '[] = '[]
  WindowSpatialShape (size ': sizes) (kernel ': kernels) (stride ': strides) (before ': after ': padding) (dilation ': dilations) = WindowExtent size kernel before after stride dilation ': WindowSpatialShape sizes kernels strides padding dilations
  WindowSpatialShape input kernel stride padding dilation = TypeError ('Text "Window kernel, stride, padding and dilation ranks must match the spatial rank")

-- | Limit pooling padding on each side to half the undilated kernel size.
type family PoolPadding (before :: Ordering) (after :: Ordering) (result :: Nat) :: Nat where
  PoolPadding 'GT after result = TypeError ('Text "Pooling padding must not exceed half the kernel size")
  PoolPadding before 'GT result = TypeError ('Text "Pooling padding must not exceed half the kernel size")
  PoolPadding before after result = result

type family PoolExtent (size :: Nat) (kernel :: Nat) (before :: Nat) (after :: Nat) (stride :: Nat) (dilation :: Nat) :: Nat where
  PoolExtent 0 kernel before after stride dilation = TypeError ('Text "Pooling spatial input dimensions must be positive")
  PoolExtent size kernel before after stride dilation = PoolPadding (CmpNat before (Div kernel 2)) (CmpNat after (Div kernel 2)) (WindowExtent size kernel before after stride dilation)

-- | Pooling validates the input and padding independently on every axis.
type family PoolSpatialShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  PoolSpatialShape '[] '[] '[] '[] '[] = '[]
  PoolSpatialShape (size ': sizes) (kernel ': kernels) (stride ': strides) (before ': after ': padding) (dilation ': dilations) = PoolExtent size kernel before after stride dilation ': PoolSpatialShape sizes kernels strides padding dilations
  PoolSpatialShape input kernel stride padding dilation = TypeError ('Text "Pooling spatial parameter ranks must match")

-- | Collapse spatial window axes into the public patch matrix layout.
type family PatchesShape (batch :: Nat) (channels :: Nat) (spatial :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  PatchesShape batch channels spatial kernel stride padding dilation = '[batch, channels * ShapeSize kernel, ShapeSize (WindowSpatialShape spatial kernel stride padding dilation)]

-- | 1D patch shape @[batch, channels * kernel_volume, positions]@ in row-major order.
type family Patches1dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  Patches1dShape '[batch, channels, size] '[k] '[s] '[left, right] '[d] = PatchesShape batch channels '[size] '[k] '[s] '[left, right] '[d]
  Patches1dShape input kernel stride padding dilation = TypeError ('Text "Patches1d requires NCL input and matching kernel, stride, padding and dilation ranks")

-- | 1D NCL pooling output with floor rounding and at most half-kernel padding per side.
type family Pool1dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  Pool1dShape '[batch, channels, size] '[k] '[s] '[left, right] '[d] = batch ': channels ': PoolSpatialShape '[size] '[k] '[s] '[left, right] '[d]
  Pool1dShape input kernel stride padding dilation = TypeError ('Text "Pool1d requires NCL input and matching kernel, stride, padding and dilation ranks")

-- | 2D patch shape @[batch, channels * kernel_volume, positions]@ in row-major order.
type family Patches2dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  Patches2dShape '[batch, channels, height, width] '[kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx] = PatchesShape batch channels '[height, width] '[kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx]
  Patches2dShape input kernel stride padding dilation = TypeError ('Text "Patches2d requires NCHW input and matching kernel, stride, padding and dilation ranks")

-- | 2D NCHW pooling output with floor rounding and at most half-kernel padding per side.
type family Pool2dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  Pool2dShape '[batch, channels, height, width] '[kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx] = batch ': channels ': PoolSpatialShape '[height, width] '[kh, kw] '[sy, sx] '[top, bottom, left, right] '[dy, dx]
  Pool2dShape input kernel stride padding dilation = TypeError ('Text "Pool2d requires NCHW input and matching kernel, stride, padding and dilation ranks")

-- | 3D patch shape @[batch, channels * kernel_volume, positions]@ in row-major order.
type family Patches3dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  Patches3dShape '[batch, channels, depth, height, width] '[kd, kh, kw] '[sz, sy, sx] '[front, back, top, bottom, left, right] '[dz, dy, dx] = PatchesShape batch channels '[depth, height, width] '[kd, kh, kw] '[sz, sy, sx] '[front, back, top, bottom, left, right] '[dz, dy, dx]
  Patches3dShape input kernel stride padding dilation = TypeError ('Text "Patches3d requires NCDHW input and matching kernel, stride, padding and dilation ranks")

-- | 3D NCDHW pooling output with floor rounding and at most half-kernel padding per side.
type family Pool3dShape (input :: [Nat]) (kernel :: [Nat]) (stride :: [Nat]) (padding :: [Nat]) (dilation :: [Nat]) :: [Nat] where
  Pool3dShape '[batch, channels, depth, height, width] '[kd, kh, kw] '[sz, sy, sx] '[front, back, top, bottom, left, right] '[dz, dy, dx] = batch ': channels ': PoolSpatialShape '[depth, height, width] '[kd, kh, kw] '[sz, sy, sx] '[front, back, top, bottom, left, right] '[dz, dy, dx]
  Pool3dShape input kernel stride padding dilation = TypeError ('Text "Pool3d requires NCDHW input and matching kernel, stride, padding and dilation ranks")
