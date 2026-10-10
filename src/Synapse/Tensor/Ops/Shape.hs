{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Shape-changing tensor operations.
module Synapse.Tensor.Ops.Shape
  ( broadcast,
    reshape,
    squeeze,
    unsqueeze,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Axes (CanSqueeze, InsertAxis, RemoveAxis)
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions
  ( Shape,
    ShapeToDIM,
    backendShape,
    shape,
    strides,
  )
import Synapse.Tensor.Shapes.Reshape (CanReshape)
import Synapse.Tensor.Type (Tensor (..))

-- | Change the shape without changing element order.
-- Callers must establish equal element counts before using this internal implementation.
reshapeTo ::
  forall to from a.
  (Shape from, Shape to, DType a) =>
  Tensor from a ->
  Tensor to a
reshapeTo (Tensor acc) = Tensor $ A.reshape (A.constant (backendShape (Proxy @to))) acc

-- | Reshape a tensor without changing its row-major element order.
-- The source and target must have equal element counts, checked at compile time.
reshape ::
  forall to from a.
  (Shape from, Shape to, CanReshape from to, DType a) =>
  Tensor from a ->
  Tensor to a
reshape = reshapeTo @to

-- | Remove a specified singleton axis.
-- Axis numbering starts at zero on the left.
-- Other singleton axes remain intact; non-singleton axes are rejected.
squeeze ::
  forall axis sh a.
  (Shape sh, Shape (RemoveAxis axis sh), CanSqueeze axis sh, DType a) =>
  Tensor sh a ->
  Tensor (RemoveAxis axis sh) a
squeeze = reshapeTo @(RemoveAxis axis sh)

-- | Insert a singleton dimension.
-- Positions range from zero through the rank, inclusive, so insertion before the first or after the last axis is permitted.
unsqueeze ::
  forall axis sh a.
  (Shape sh, Shape (InsertAxis axis 1 sh), DType a) =>
  Tensor sh a ->
  Tensor (InsertAxis axis 1 sh) a
unsqueeze = reshapeTo @(InsertAxis axis 1 sh)

-- | Broadcast a tensor to a target shape.
--
-- The target shape is usually inferred from context.
-- When it is ambiguous, use type application:
--
-- @
-- broadcast @'[32, 128] bias
-- @
broadcast ::
  forall to from a.
  (Shape from, Shape to, CanBroadcast from to, DType a) =>
  Tensor from a ->
  Tensor to a
broadcast (Tensor acc) =
  Tensor $
    A.backpermute
      (A.constant (backendShape (Proxy @to)))
      sourceIndex
      acc
  where
    sourceIndex :: A.Exp (ShapeToDIM to) -> A.Exp (ShapeToDIM from)
    sourceIndex ix =
      A.fromIndex
        (A.constant (backendShape (Proxy @from)))
        (sourceLinearIndex ix)

    -- \| Map target indices to source elements; singleton source axes always contribute coordinate zero, and leading target axes do not contribute.
    sourceLinearIndex :: A.Exp (ShapeToDIM to) -> A.Exp Int
    sourceLinearIndex ix =
      foldr addAxis 0 axisMappings
      where
        targetLinear = A.toIndex (A.constant (backendShape (Proxy @to))) ix

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
