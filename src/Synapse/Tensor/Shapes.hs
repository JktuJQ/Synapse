-- | Tensor shape types, runtime dimensions, and operation constraints.
-- Shapes use left-to-right axis order, and each operation checks its required shape relationships at compile time.
module Synapse.Tensor.Shapes
  ( ShapeToDIM,
    ShapeSize,
    Shape (..),
    CanBroadcast,
    CanReshape,
    Rank,
    AxisSize,
    RemoveAxis,
    InsertAxis,
    SelectAxes,
    CanSelectAxes,
    CanSqueeze,
    ReduceShape,
    CanReduce,
    CanReduceNonEmpty,
    NonEmpty,
    Axes (..),
    Index,
    Slice,
    SliceAxis (..),
    CanIndex,
    CanSlice,
    SliceShape,
    MatmulShape,
    Conv1dShape,
    Conv2dShape,
    Conv3dShape,
    ConvTranspose1dShape,
    ConvTranspose2dShape,
    ConvTranspose3dShape,
    Patches1dShape,
    Patches2dShape,
    Patches3dShape,
    Pool1dShape,
    Pool2dShape,
    Pool3dShape,
    PermuteAxes,
    TransposeShape,
    ConcatShape,
  )
where

import Synapse.Tensor.Shapes.Axes
  ( Axes (..),
    AxisSize,
    CanReduce,
    CanReduceNonEmpty,
    CanSelectAxes,
    CanSqueeze,
    ConcatShape,
    InsertAxis,
    NonEmpty,
    PermuteAxes,
    Rank,
    ReduceShape,
    RemoveAxis,
    SelectAxes,
    TransposeShape,
  )
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Convolution (Conv1dShape, Conv2dShape, Conv3dShape, ConvTranspose1dShape, ConvTranspose2dShape, ConvTranspose3dShape)
import Synapse.Tensor.Shapes.Dimensions (Shape (..), ShapeSize, ShapeToDIM)
import Synapse.Tensor.Shapes.Index (CanIndex, CanSlice, Index, Slice, SliceAxis (..), SliceShape)
import Synapse.Tensor.Shapes.Matmul (MatmulShape)
import Synapse.Tensor.Shapes.Reshape (CanReshape)
import Synapse.Tensor.Shapes.Window (Patches1dShape, Patches2dShape, Patches3dShape, Pool1dShape, Pool2dShape, Pool3dShape)
