{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Differentiable extraction and overlap summation of spatial patches.
module Synapse.Tensor.Ops.Window (extractPatches1d, extractPatches2d, extractPatches3d, combinePatches1d, combinePatches2d, combinePatches3d) where

import qualified Synapse.Autograd.Tensor.Ops.Primitives.Window as Pullback
import Synapse.Tensor.Ops (TensorOps (lift1))
import qualified Synapse.Tensor.Ops.Primitives.Window as Raw
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Window (Patches1dShape, Patches2dShape, Patches3dShape)
import Synapse.Tensor.Type (TensorElement)

-- | Extract zero-padded 1D windows from NCL input as @[batch, channels * kernel_volume, positions]@.
-- Type arguments specify one kernel size, one stride and @[left, right]@ padding, followed by one positive dilation per spatial axis.
-- Kernel coordinates and output positions follow row-major order.
extractPatches1d :: forall kernel stride padding dilation {t} input a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape (Patches1dShape input kernel stride padding dilation)) => t input a -> t (Patches1dShape input kernel stride padding dilation) a
extractPatches1d = lift1 (Raw.extractPatches @kernel @stride @padding @dilation) (Pullback.extractPatches @kernel @stride @padding @dilation)

-- | Extract zero-padded 2D windows from NCHW input as @[batch, channels * kernel_volume, positions]@.
-- Type arguments specify 2 kernel sizes, 2 strides and @[top, bottom, left, right]@ padding, followed by one positive dilation per spatial axis.
-- Kernel coordinates and output positions follow row-major order.
extractPatches2d :: forall kernel stride padding dilation {t} input a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape (Patches2dShape input kernel stride padding dilation)) => t input a -> t (Patches2dShape input kernel stride padding dilation) a
extractPatches2d = lift1 (Raw.extractPatches @kernel @stride @padding @dilation) (Pullback.extractPatches @kernel @stride @padding @dilation)

-- | Extract zero-padded 3D windows from NCDHW input as @[batch, channels * kernel_volume, positions]@.
-- Type arguments specify 3 kernel sizes, 3 strides and @[front, back, top, bottom, left, right]@ padding, followed by one positive dilation per spatial axis.
-- Kernel coordinates and output positions follow row-major order.
extractPatches3d :: forall kernel stride padding dilation {t} input a. (TensorOps t a, Num (TensorElement a), Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape (Patches3dShape input kernel stride padding dilation)) => t input a -> t (Patches3dShape input kernel stride padding dilation) a
extractPatches3d = lift1 (Raw.extractPatches @kernel @stride @padding @dilation) (Pullback.extractPatches @kernel @stride @padding @dilation)

-- | Sum patches into a specified NCL output shape, dropping entries outside that shape.
-- Type arguments specify the full output shape, kernel, stride and padding, followed by one positive dilation per spatial axis.
-- Overlaps are summed and uncovered elements are zero; the patch shape must match extraction from that output.
combinePatches1d :: forall output kernel stride padding dilation {t} patches a. (TensorOps t a, Num (TensorElement a), Shape output, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape patches, patches ~ Patches1dShape output kernel stride padding dilation) => t patches a -> t output a
combinePatches1d = lift1 (Raw.combinePatches @output @kernel @stride @padding @dilation) (Pullback.combinePatches @kernel @stride @padding @dilation)

-- | Sum patches into a specified NCHW output shape, dropping entries outside that shape.
-- Type arguments specify the full output shape, kernel, stride and padding, followed by one positive dilation per spatial axis.
-- Overlaps are summed and uncovered elements are zero; the patch shape must match extraction from that output.
combinePatches2d :: forall output kernel stride padding dilation {t} patches a. (TensorOps t a, Num (TensorElement a), Shape output, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape patches, patches ~ Patches2dShape output kernel stride padding dilation) => t patches a -> t output a
combinePatches2d = lift1 (Raw.combinePatches @output @kernel @stride @padding @dilation) (Pullback.combinePatches @kernel @stride @padding @dilation)

-- | Sum patches into a specified NCDHW output shape, dropping entries outside that shape.
-- Type arguments specify the full output shape, kernel, stride and padding, followed by one positive dilation per spatial axis.
-- Overlaps are summed and uncovered elements are zero; the patch shape must match extraction from that output.
combinePatches3d :: forall output kernel stride padding dilation {t} patches a. (TensorOps t a, Num (TensorElement a), Shape output, Shape kernel, Shape stride, Shape padding, Shape dilation, Shape patches, patches ~ Patches3dShape output kernel stride padding dilation) => t patches a -> t output a
combinePatches3d = lift1 (Raw.combinePatches @output @kernel @stride @padding @dilation) (Pullback.combinePatches @kernel @stride @padding @dilation)
