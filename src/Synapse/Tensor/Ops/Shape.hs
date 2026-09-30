{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Shape-changing tensor operations.
module Synapse.Tensor.Ops.Shape
  ( broadcast,
    reshape,
  )
where

import Data.Proxy (Proxy (..))

import qualified Data.Array.Accelerate as A

import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shape
  ( CanBroadcast,
    CanReshape,
    KnownShape,
    ShapeToDIM,
    shape,
    shapeVal,
  )
import Synapse.Tensor.Type (Tensor (..))

-- | Reshape a tensor without changing its row-major element order.
reshape ::
  forall to from a.
  (KnownShape from, KnownShape to, CanReshape from to, DType a) =>
  Tensor from a ->
  Tensor to a
reshape (Tensor acc) =
  Tensor $
    A.backpermute
      targetShape
      sourceIndex
      acc
  where
    sourceShape = A.constant (shapeVal (Proxy @from))
    targetShape = A.constant (shapeVal (Proxy @to))

    sourceIndex :: A.Exp (ShapeToDIM to) -> A.Exp (ShapeToDIM from)
    sourceIndex ix =
      A.fromIndex sourceShape (A.toIndex targetShape ix)

-- | Broadcast a tensor to a target shape.
--
-- The target shape is usually inferred from context. When it is ambiguous, use
-- type application:
--
-- @
-- broadcast @'[32, 128] bias
-- @
broadcast ::
  forall to from a.
  (KnownShape from, KnownShape to, CanBroadcast from to, DType a) =>
  Tensor from a ->
  Tensor to a
broadcast (Tensor acc) =
  Tensor $
    A.backpermute
      (A.constant (shapeVal (Proxy @to)))
      sourceIndex
      acc
  where
    sourceIndex :: A.Exp (ShapeToDIM to) -> A.Exp (ShapeToDIM from)
    sourceIndex ix =
      A.fromIndex
        (A.constant (shapeVal (Proxy @from)))
        (sourceLinearIndex ix)

    -- Compute the source linear index corresponding to a target index.
    sourceLinearIndex :: A.Exp (ShapeToDIM to) -> A.Exp Int
    sourceLinearIndex ix =
      foldr addAxis 0 axisMappings
      where
        targetLinear = A.toIndex (A.constant (shapeVal (Proxy @to))) ix

        addAxis :: (Int, Int, (Int, Int)) -> A.Exp Int -> A.Exp Int
        addAxis (fromDim, toDim, (fromStride, toStride)) linearAcc
          | fromDim == 1 = linearAcc
          | otherwise =
              let coord = (targetLinear `quot` A.constant toStride) `rem` A.constant toDim
               in coord * A.constant fromStride + linearAcc

    fromDims = shape @from
    toDims = shape @to
    alignedToDims = drop (length toDims - length fromDims) toDims
    fromStrides = strides fromDims
    toStrides = strides toDims
    alignedToStrides = drop (length toStrides - length fromDims) toStrides
    axisMappings = zip3 fromDims alignedToDims (zip fromStrides alignedToStrides)

    -- Row-major strides for a shape.
    strides :: [Int] -> [Int]
    strides [] = []
    strides (_ : dims) = product dims : strides dims
