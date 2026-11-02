{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Shape-changing tensor operations.
module Synapse.Tensor.Ops.Shape (broadcast, reshape, squeeze, unsqueeze, transpose, permuteAxes, concat, stack) where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat)
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Shape as Pullback
import Synapse.Tensor.Ops (TensorOps (..))
import qualified Synapse.Tensor.Ops.Primitives.Shape as Raw
import Synapse.Tensor.Shapes.Axes (Axes, CanSqueeze, ConcatShape, InsertAxis, PermuteAxes, RemoveAxis, TransposeShape)
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Reshape (CanReshape)
import Prelude hiding (concat)

-- | Reshape a tensor without changing its row-major element order.
-- The source and target must have equal element counts, checked at compile time.
reshape ::
  forall {t} to from a.
  (TensorOps t a, Shape from, Shape to, CanReshape from to) =>
  t from a ->
  t to a
reshape = lift1 (Raw.reshape @to) Pullback.reshape

-- | Remove a specified singleton axis.
-- Axis numbering starts at zero on the left.
-- Other singleton axes remain intact; non-singleton axes are rejected.
squeeze ::
  forall {t} axis sh a.
  (TensorOps t a, Shape sh, Shape (RemoveAxis axis sh), CanSqueeze axis sh) =>
  t sh a ->
  t (RemoveAxis axis sh) a
squeeze = lift1 (Raw.reshape @(RemoveAxis axis sh)) Pullback.reshape

-- | Insert a singleton dimension.
-- Positions range from zero through the rank, inclusive, so insertion before the first or after the last axis is permitted.
unsqueeze ::
  forall {t} axis sh a.
  (TensorOps t a, Shape sh, Shape (InsertAxis axis 1 sh)) =>
  t sh a ->
  t (InsertAxis axis 1 sh) a
unsqueeze = lift1 (Raw.reshape @(InsertAxis axis 1 sh)) Pullback.reshape

-- | Reorder axes so each output axis corresponds to the specified input axis.
-- For example, @permuteAxes \@'[1, 2, 0]@ maps @'[a, b, c]@ to @'[b, c, a]@.
-- The order must contain every axis exactly once and scalars accept an empty order.
permuteAxes ::
  forall {t} xs sh a.
  (TensorOps t a, Axes xs, Shape sh, Shape (PermuteAxes xs sh)) =>
  t sh a ->
  t (PermuteAxes xs sh) a
permuteAxes = lift1 (Raw.permuteAxes @xs Proxy) (Pullback.permuteAxes @xs Proxy)

-- | Transpose the last two axes of a matrix or batched matrix.
-- Leading batch axes retain their order and inputs must have rank at least two.
transpose ::
  forall {t} sh a.
  (TensorOps t a, Shape sh, Shape (TransposeShape sh)) =>
  t sh a ->
  t (TransposeShape sh) a
transpose = lift1 Raw.transpose Pullback.transpose

-- | Concatenate two tensors along an existing zero-based axis.
-- Both ranks and all dimensions outside that axis must match.
concat ::
  forall {t} axis lhs rhs a.
  (TensorOps t a, KnownNat axis, Shape lhs, Shape rhs, Shape (ConcatShape axis lhs rhs)) =>
  t lhs a ->
  t rhs a ->
  t (ConcatShape axis lhs rhs) a
concat = lift2 (Raw.concat @axis Proxy) (Pullback.concat @axis Proxy)

-- | Stack two equally shaped tensors by inserting a dimension of size two.
-- Positions range from zero through the input rank, so scalars may be stacked at position zero.
stack ::
  forall {t} axis sh a.
  (TensorOps t a, KnownNat axis, Shape sh, Shape (InsertAxis axis 2 sh)) =>
  t sh a ->
  t sh a ->
  t (InsertAxis axis 2 sh) a
stack = lift2 (Raw.stack @axis Proxy) (Pullback.stack @axis Proxy)

-- | Broadcast a tensor to a target shape.
--
-- The target shape is usually inferred from context.
-- When it is ambiguous, use type application:
--
-- @
-- broadcast @'[32, 128] bias
-- @
broadcast ::
  forall {t} to from a.
  (TensorOps t a, Shape from, Shape to, CanBroadcast from to) =>
  t from a ->
  t to a
broadcast = lift1 (Raw.broadcast @to) Pullback.broadcast
