-- | Tensor shapes, runtime dimensions, and type-level operation constraints.
module Synapse.Tensor.Shapes
    ( -- * Dimensions
      Shape (..)
    , ShapeSize
    , ShapeToDIM
    , Rank

      -- * Broadcasting and reshaping
    , CanBroadcast
    , CanReshape

      -- * Axis transformations
    , Axes (..)
    , AxisSize
    , RemoveAxis
    , InsertAxis
    , SelectAxes
    , CanSelectAxes
    , CanSqueeze
    , PermuteAxes
    , TransposeShape
    , ConcatShape

      -- * Reduction
    , ReduceShape
    , CanReduce
    , CanReduceNonEmpty
    , NonEmpty

      -- * Indexing and slicing
    , Index
    , CanIndex
    , Slice
    , SliceAxis (..)
    , CanSlice
    , SliceShape
    , LookupRowsShape

      -- * Matrix multiplication
    , MatmulShape

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

      -- * Normalization
    , LayerNormAxes
    , BatchNormAxes
    )
where

import Synapse.Tensor.Shapes.Axes
    ( Axes (..)
    , AxisSize
    , CanReduce
    , CanReduceNonEmpty
    , CanSelectAxes
    , CanSqueeze
    , ConcatShape
    , InsertAxis
    , NonEmpty
    , PermuteAxes
    , Rank
    , ReduceShape
    , RemoveAxis
    , SelectAxes
    , TransposeShape
    )
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape (..), ShapeSize, ShapeToDIM)
import Synapse.Tensor.Shapes.Index
    ( CanIndex
    , CanSlice
    , Index
    , LookupRowsShape
    , Slice
    , SliceAxis (..)
    , SliceShape
    )
import Synapse.Tensor.Shapes.Matmul (MatmulShape)
import Synapse.Tensor.Shapes.Normalization (BatchNormAxes, LayerNormAxes)
import Synapse.Tensor.Shapes.Reshape (CanReshape)
import Synapse.Tensor.Shapes.Window
    ( Conv1dShape
    , Conv2dShape
    , Conv3dShape
    , ConvTranspose1dShape
    , ConvTranspose2dShape
    , ConvTranspose3dShape
    , Patches1dShape
    , Patches2dShape
    , Patches3dShape
    , Pool1dShape
    , Pool2dShape
    , Pool3dShape
    )
