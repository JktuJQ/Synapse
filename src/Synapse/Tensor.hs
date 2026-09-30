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
    shape,
    shapeSize,
    scalar,
    toList,
    unsafeFromList,
  )
where

import Synapse.Tensor.Backend (Backend (..), TensorValue, run, runCPU, runGPU, toList)
import Synapse.Tensor.Core (Tensor, fromList, scalar, unsafeFromList)
import Synapse.Tensor.DType (DType, Float32, Float64)
import Synapse.Tensor.Shape (KnownShape, shape, shapeSize)
