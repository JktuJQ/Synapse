-- | Tensor arithmetic, indexing, shape transformations, reductions, and random sampling.
module Synapse.Tensor.Ops
    ( TensorOps (..)
    , withVJP
    , withCachedVJP
    , module Synapse.Tensor.Ops.Construction
    , module Synapse.Tensor.Ops.Elementwise
    , module Synapse.Tensor.Ops.Index
    , module Synapse.Tensor.Ops.Shape
    , module Synapse.Tensor.Ops.Reduction
    , module Synapse.Tensor.Ops.Math
    , module Synapse.Tensor.Ops.Random
    , module Synapse.Tensor.Ops.Window
    ) where

import Synapse.Tensor.Ops.Construction
import Synapse.Tensor.Ops.Definition (TensorOps (..), withCachedVJP, withVJP)
import Synapse.Tensor.Ops.Elementwise
import Synapse.Tensor.Ops.Index
import Synapse.Tensor.Ops.Math
import Synapse.Tensor.Ops.Random
import Synapse.Tensor.Ops.Reduction
import Synapse.Tensor.Ops.Shape
import Synapse.Tensor.Ops.Window
