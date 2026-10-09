-- | Tensor shape types, runtime dimensions, and operation constraints.
--
-- Definitions live in the corresponding Shapes submodules. This module keeps
-- the shared shape interface available through a single import.
module Synapse.Tensor.Shapes
  ( ShapeToDIM,
    ShapeSize,
    Shape,
    CanBroadcast,
    CanReshape,
    shapeVal,
    shapeList,
    shape,
    shapeSize,
  )
where

import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeSize, ShapeToDIM, shape, shapeList, shapeSize, shapeVal)
import Synapse.Tensor.Shapes.Reshape (CanReshape)
