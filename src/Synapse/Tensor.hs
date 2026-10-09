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
    Shape,
    arange,
    eye,
    fill,
    fromGen,
    fromList,
    fromData,
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
    shapeRank,
    scalar,
    toList,
    unsafeFromList,
    zeros,
    zerosLike,
    broadcast,
    squeeze,
    unsqueeze,
    permuteAxes,
  )
where

import Synapse.Tensor.Backend (Backend (..), run, runCPU, runGPU)
import Synapse.Tensor.Construction
  ( arange,
    eye,
    fill,
    fromGen,
    full,
    fullLike,
    identity,
    linspace,
    logspace,
    ones,
    onesLike,
    scalar,
    zeros,
    zerosLike,
  )
import Synapse.Tensor.Conversion (fromData, fromList, toList, unsafeFromList)
import Synapse.Tensor.DType (DType, Float32, Float64)
import Synapse.Tensor.Ops.Axes (permuteAxes, squeeze, unsqueeze)
import Synapse.Tensor.Ops.Elementwise (map, zip, zipWith)
import Synapse.Tensor.Ops.Math ()
import Synapse.Tensor.Ops.Shape (broadcast, reshape)
import Synapse.Tensor.Shapes (Shape, shape, shapeRank, shapeSize)
import Synapse.Tensor.Type (FromIntegral, Tensor, TensorData, TensorElement, constant, fromIntegral)
import Prelude hiding (fromIntegral, map, zip, zipWith)
