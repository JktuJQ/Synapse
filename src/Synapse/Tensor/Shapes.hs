-- | Tensor shape types, runtime dimensions, and operation constraints.
--
-- Definitions live in the corresponding Shapes submodules.
-- This module keeps the shared shape interface available through a single import.
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
    Axes (..),
    Index,
    Slice,
    SliceAxis (..),
    CanIndex,
    CanSlice,
    SliceShape,
  )
where

import Synapse.Tensor.Shapes.Axes
  ( Axes (..),
    AxisSize,
    CanSelectAxes,
    CanSqueeze,
    InsertAxis,
    Rank,
    RemoveAxis,
    SelectAxes,
  )
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape (..), ShapeSize, ShapeToDIM)
import Synapse.Tensor.Shapes.Index (CanIndex, CanSlice, Index, Slice, SliceAxis (..), SliceShape)
import Synapse.Tensor.Shapes.Reshape (CanReshape)
