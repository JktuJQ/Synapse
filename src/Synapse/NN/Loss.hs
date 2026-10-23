{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Loss functions with caller-supplied reductions.
module Synapse.NN.Loss (mse, mae, huber, smoothL1) where

import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Type (TensorLike)

-- | Compute squared differences between predictions and targets, then apply a reduction.
-- Both inputs must have the same shape; broadcasting is explicit.
-- The reduction determines the output shape and the treatment of empty inputs.
-- Ordinary and differentiable tensors use the same function.
--
-- Using operations from "Synapse.Tensor":
--
-- @
-- mse T.mean predictions targets -- Mean squared error.
-- mse T.sum predictions targets  -- Sum of squared errors.
-- mse id predictions targets     -- Elementwise squared errors.
-- @
mse ::
  forall {t :: TensorLike} in_sh out_sh a.
  (Num (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
mse reduce predictions targets =
  let difference = predictions - targets
   in reduce $ difference * difference

-- | Compute absolute differences, then apply a reduction as in 'mse'.
-- The derivative at equal predictions and targets is zero.
mae ::
  forall {t :: TensorLike} in_sh out_sh a.
  (Num (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
mae reduce predictions targets = reduce $ abs $ predictions - targets

-- | Compute Huber errors with a finite, positive delta, then apply a reduction.
-- For an absolute difference r, the error is r² / 2 below delta and delta * (r - delta / 2) otherwise.
-- Invalid delta values raise an error before constructing the loss.
-- For example, @huber T.mean 1 predictions targets@ uses a transition at one.
huber ::
  forall {t :: TensorLike} in_sh out_sh a.
  (RealFloat a, Fractional (t in_sh a), E.Ord (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  a ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
huber reduce delta predictions targets
  | isNaN delta || isInfinite delta || delta <= 0 = error "Synapse.NN.huber: delta must be finite and positive"
  | otherwise =
      let threshold = realToFrac delta
          magnitude = abs $ predictions - targets
          quadratic = E.min magnitude threshold
       in reduce $ (0.5 * quadratic) * quadratic + threshold * (magnitude - quadratic)

-- | Compute Smooth L1 errors with a finite, nonnegative beta, then apply a reduction.
-- For an absolute difference r, the error is r² / (2 * beta) below beta and r - beta / 2 otherwise.
-- A zero beta gives 'mae'; negative or nonfinite beta values raise an error.
-- For positive beta, this is Huber loss divided by beta.
smoothL1 ::
  forall {t :: TensorLike} in_sh out_sh a.
  (RealFloat a, Fractional (t in_sh a), E.Ord (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  a ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
smoothL1 reduce beta predictions targets
  | isNaN beta || isInfinite beta || beta < 0 = error "Synapse.NN.smoothL1: beta must be finite and nonnegative"
  | beta == 0 = mae reduce predictions targets
  | otherwise =
      let threshold = realToFrac beta
          magnitude = abs $ predictions - targets
          quadratic = E.min magnitude threshold
       in reduce $ (0.5 * quadratic) * (quadratic / threshold) + (magnitude - quadratic)
