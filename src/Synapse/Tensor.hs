-- | Public tensor API.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.Tensor as T
-- @
--
-- 'Tensor' is an array computation, 'TensorElement' an element expression, and 'TensorData' a materialized array.
module Synapse.Tensor
  ( Tensor,
    TensorData,
    TensorElement,
    TensorIndex,
    coordinate,
    Indexable,
    (!!),
    FromIntegral,
    Eq,
    Ord,
    (==),
    (/=),
    (<),
    (<=),
    (>),
    (>=),
    min,
    max,
    constant,
    fromIntegral,
    Backend (..),
    DType,
    Float32,
    Float64,
    Shape,
    SliceAxis (..),
    index,
    slice,
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
    sum,
    sumAxes,
    mean,
    meanAxes,
    minimum,
    minimumAxes,
    maximum,
    maximumAxes,
    dot,
    matmul,
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
import Synapse.Tensor.Ops.Elementwise (Eq, Ord, map, max, min, zip, zipWith, (/=), (<), (<=), (==), (>), (>=))
import Synapse.Tensor.Ops.Index (Indexable, coordinate, index, slice, (!!))
import Synapse.Tensor.Ops.Math (dot, matmul)
import Synapse.Tensor.Ops.Reduction (maximum, maximumAxes, mean, meanAxes, minimum, minimumAxes, sum, sumAxes)
import Synapse.Tensor.Ops.Shape (broadcast, reshape, squeeze, unsqueeze)
import Synapse.Tensor.Shapes (Shape, shape, shapeRank, shapeSize)
import Synapse.Tensor.Shapes.Index (SliceAxis (..))
import Synapse.Tensor.Type (FromIntegral, Tensor, TensorData, TensorElement, TensorIndex, constant, fromIntegral)
import Prelude hiding (Eq, Ord, fromIntegral, map, max, maximum, min, minimum, sum, zip, zipWith, (!!), (/=), (<), (<=), (==), (>), (>=))
