-- | Tensor shape types, runtime dimensions, and operation constraints.
--
-- Definitions live in the corresponding Shapes submodules. This module keeps
-- the shared shape interface available through a single import.
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
    PermuteAxes,
    CanSelectAxes,
    CanPermuteAxes,
    CanSqueeze,
    Axes (..),
  )
where

import Synapse.Tensor.Shapes.Axes
  ( Axes (..),
    AxisSize,
    CanPermuteAxes,
    CanSelectAxes,
    CanSqueeze,
    InsertAxis,
    PermuteAxes,
    Rank,
    RemoveAxis,
    SelectAxes,
  )
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape (..), ShapeSize, ShapeToDIM)
import Synapse.Tensor.Shapes.Reshape (CanReshape)
