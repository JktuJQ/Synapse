-- | Tensor shapes, runtime dimensions, and type-level operation constraints.
module Synapse.Tensor.Shapes
    ( module Synapse.Tensor.Shapes.Dimensions
    , module Synapse.Tensor.Shapes.Broadcast
    , module Synapse.Tensor.Shapes.Reshape
    , module Synapse.Tensor.Shapes.Axes
    , module Synapse.Tensor.Shapes.Index
    , module Synapse.Tensor.Shapes.Matmul
    , module Synapse.Tensor.Shapes.Window
    , module Synapse.Tensor.Shapes.Normalization
    ) where

import Synapse.Tensor.Shapes.Axes
import Synapse.Tensor.Shapes.Broadcast
import Synapse.Tensor.Shapes.Dimensions
import Synapse.Tensor.Shapes.Index
import Synapse.Tensor.Shapes.Matmul
import Synapse.Tensor.Shapes.Normalization
import Synapse.Tensor.Shapes.Reshape
import Synapse.Tensor.Shapes.Window
