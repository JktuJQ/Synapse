-- | Tensor arithmetic, indexing, shape transformations, reductions, and random sampling.
module Synapse.Tensor.Ops
    ( TensorOps (..)
    , withVJP
    , withCachedVJP
    , module Synapse.Tensor.Ops.Constraints
    , module Synapse.Tensor.Ops.Construction
    , module Synapse.Tensor.Ops.Logic
    , module Synapse.Tensor.Ops.Numeric
    , module Synapse.Tensor.Ops.Conversion
    , module Synapse.Tensor.Ops.Index
    , module Synapse.Tensor.Ops.Shape
    , module Synapse.Tensor.Ops.Reduction
    , module Synapse.Tensor.Ops.Math
    , module Synapse.Tensor.Ops.Random
    , module Synapse.Tensor.Ops.Window
    ) where

import Synapse.Tensor.Ops.Constraints
import Synapse.Tensor.Ops.Construction
import Synapse.Tensor.Ops.Conversion
import Synapse.Tensor.Ops.Definition (TensorOps (..), withCachedVJP, withVJP)
import Synapse.Tensor.Ops.Index
import Synapse.Tensor.Ops.Logic
import Synapse.Tensor.Ops.Math
import Synapse.Tensor.Ops.Numeric
import Synapse.Tensor.Ops.Random
import Synapse.Tensor.Ops.Reduction
import Synapse.Tensor.Ops.Shape
import Synapse.Tensor.Ops.Window
