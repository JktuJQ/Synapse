-- | Public tensor API.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.Tensor as T
-- @
module Synapse.Tensor
  ( Tensor,
    TensorValue,
    Backend (..),
    DType,
    Float32,
    Float64,
    KnownShape,
    fromList,
    run,
    runCPU,
    runGPU,
    reshape,
    shape,
    shapeSize,
    scalar,
    toList,
    unsafeFromList,
    broadcast,
  )
where

import Synapse.Tensor.Backend (Backend (..), TensorValue, run, runCPU, runGPU, toList)
import Synapse.Tensor.Construction (fromList, scalar, unsafeFromList)
import Synapse.Tensor.DType (DType, Float32, Float64)
import Synapse.Tensor.Ops.Shape (broadcast, reshape)
import Synapse.Tensor.Shape (KnownShape, shape, shapeSize)
import Synapse.Tensor.Type (Tensor)
