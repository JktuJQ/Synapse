-- | Public tensor API.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.Tensor as T
-- @
--
-- 'Tensor' is an array computation, 'TensorElement' an element expression,
-- and 'TensorData' a materialized array.
module Synapse.Tensor
  ( Tensor,
    TensorData,
    TensorElement,
    FromIntegral,
    constant,
    fromIntegral,
    Backend (..),
    DType,
    Float32,
    Float64,
    KnownShape,
    arange,
    eye,
    fill,
    fromGen,
    fromList,
    full,
    fullLike,
    identity,
    linspace,
    logspace,
    map,
    zip,
    zipWith,
    ones,
    onesLike,
    run,
    runCPU,
    runGPU,
    reshape,
    shape,
    shapeSize,
    scalar,
    toList,
    unsafeFromList,
    zeros,
    zerosLike,
    broadcast,
  )
where

import Synapse.Tensor.Backend (Backend (..), run, runCPU, runGPU, toList)
import Synapse.Tensor.Construction
  ( arange,
    eye,
    fill,
    fromGen,
    fromList,
    full,
    fullLike,
    identity,
    linspace,
    logspace,
    ones,
    onesLike,
    scalar,
    unsafeFromList,
    zeros,
    zerosLike,
  )
import Synapse.Tensor.DType (DType, Float32, Float64)
import Synapse.Tensor.Ops.Elementwise (map, zip, zipWith)
import Synapse.Tensor.Ops.Math ()
import Synapse.Tensor.Ops.Shape (broadcast, reshape)
import Synapse.Tensor.Shape (KnownShape, shape, shapeSize)
import Synapse.Tensor.Type (FromIntegral, Tensor, TensorData, TensorElement, constant, fromIntegral)
import Prelude hiding (fromIntegral, map, zip, zipWith)
