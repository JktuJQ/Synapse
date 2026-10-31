-- | Public tensor API.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.Tensor as T
-- @
--
-- Tensor construction, shape transformations, reductions, and matrix products share the 'TensorOps' constraint.
-- The same function can be used with ordinary tensors or the wrappers from "Synapse.Autograd.Tensor".
--
-- @
-- gramTotal :: T.TensorOps t Double => t '[2, 3] Double -> t '[] Double
-- gramTotal x = T.sum (T.matmul x (T.transpose x))
-- @
module Synapse.Tensor
  ( FieldPath,
    TensorFields,
    traverseTensors,
    WithTensor,
    TensorOps,
    Element,
    Tensor,
    TensorLike,
    TensorData,
    TensorElement,
    TensorIndex,
    coordinate,
    Indexable,
    (!!),
    FromIntegral,
    Eq,
    Ord,
    Boolean,
    Integral,
    RealFloat,
    quot,
    rem,
    isNaN,
    isInfinite,
    all,
    any,
    allAxes,
    anyAxes,
    softmax,
    logSoftmax,
    logSumExp,
    logSumExpAxes,
    Conditional,
    not,
    (&&),
    (||),
    select,
    cond,
    (==),
    (/=),
    (<),
    (<=),
    (>),
    (>=),
    min,
    max,
    element,
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
    fromListData,
    fromList,
    fromData,
    full,
    fullLike,
    identity,
    linspace,
    logspace,
    ones,
    onesLike,
    run,
    run1,
    run2,
    run3,
    run4,
    run5,
    run6,
    run7,
    run8,
    run9,
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
    argmax,
    argmin,
    dot,
    matmul,
    transpose,
    permuteAxes,
    concat,
    stack,
  )
where

import Synapse.Tensor.Backend (Backend (..), run, run1, run2, run3, run4, run5, run6, run7, run8, run9)
import Synapse.Tensor.Conversion (fromData, fromList, fromListData, toList, unsafeFromList)
import Synapse.Tensor.Fields (FieldPath, TensorFields (traverseTensors))
import Synapse.Tensor.Ops (TensorOps)
import Synapse.Tensor.Ops.Construction
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
import Synapse.Tensor.Ops.Elementwise (Boolean, Conditional, Eq, Integral, Ord, RealFloat, cond, isInfinite, isNaN, max, min, not, quot, rem, select, (&&), (/=), (<), (<=), (==), (>), (>=), (||))
import Synapse.Tensor.Ops.Index (Indexable, coordinate, index, slice, (!!))
import Synapse.Tensor.Ops.Math (dot, matmul)
import Synapse.Tensor.Ops.Reduction (all, allAxes, any, anyAxes, argmax, argmin, logSoftmax, logSumExp, logSumExpAxes, maximum, maximumAxes, mean, meanAxes, minimum, minimumAxes, softmax, sum, sumAxes)
import Synapse.Tensor.Ops.Shape (broadcast, concat, permuteAxes, reshape, squeeze, stack, transpose, unsqueeze)
import Synapse.Tensor.Shapes (Shape, shape, shapeRank, shapeSize)
import Synapse.Tensor.Shapes.Index (SliceAxis (..))
import Synapse.Tensor.Type (DType, Element, Float32, Float64, FromIntegral, Tensor, TensorData, TensorElement, TensorIndex, TensorLike, WithTensor, element, fromIntegral)
import Prelude hiding (Eq, Integral, Ord, RealFloat, all, any, concat, fromIntegral, isInfinite, isNaN, max, maximum, min, minimum, not, quot, rem, sum, (!!), (&&), (/=), (<), (<=), (==), (>), (>=), (||))
