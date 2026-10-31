{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Loss functions with caller-supplied reductions.
module Synapse.NN.Loss (mse, mae, huber, smoothL1, binaryCrossEntropy, binaryCrossEntropyWithLogits, negativeLogLikelihood, crossEntropy, unsafeNegativeLogLikelihood, unsafeCrossEntropy, smoothLabels, unsafeCrossEntropySmoothed) where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import qualified Synapse.Autograd.Tensor.Primitives.Index as Pullback
import Synapse.NN.Activation (softplus)
import Synapse.Tensor.Ops (TensorOps (lift1, lift2))
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Reduction as R
import qualified Synapse.Tensor.Primitives.Index as Raw
import Synapse.Tensor.Shapes.Axes (Axes, CanReduceNonEmpty, ReduceShape, RemoveAxis)
import Synapse.Tensor.Shapes.Dimensions (Shape, shape)
import Synapse.Tensor.Type (Tensor, TensorElement, TensorLike)

-- | Reflect the size of an axis whose validity is established by the calling loss.
classCount :: forall axis sh. (KnownNat axis, Shape sh) => Int
classCount = shape @sh !! fromInteger (toInteger $ natVal $ Proxy @axis)

-- | Mix class probabilities with a uniform distribution: @(1 - smoothing) * targets + smoothing / classes@.
-- Smoothing must lie in @[0, 1]@ and the class axis must be nonempty.
-- Targets retain their shape and differentiation history.
smoothLabels ::
  forall {t :: TensorLike} axis sh a.
  (KnownNat axis, Shape sh, CanReduceNonEmpty '[axis] sh, RealFloat a, Fractional (t sh a)) =>
  a ->
  t sh a ->
  t sh a
smoothLabels smoothing targets =
  let classes = classCount @axis @sh
   in realToFrac (1 - smoothing) * targets + realToFrac (smoothing / fromIntegral classes)

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

-- | Binary cross entropy @-(y * log p + (1 - y) * log (1 - p))@ from probabilities.
-- Predictions and targets have the same shape and must lie in @[0, 1]@.
-- Logarithms are bounded below by -100 so endpoint probabilities have finite losses, as in PyTorch's BCELoss.
-- The reduction is supplied explicitly, as in 'mse'.
-- Prefer 'binaryCrossEntropyWithLogits' when the model returns unnormalized scores.
binaryCrossEntropy ::
  forall {t :: TensorLike} sh out_sh a.
  (TensorOps t a, Shape sh, RealFloat a, Floating (TensorElement a), E.Ord (TensorElement a)) =>
  (t sh a -> t out_sh a) ->
  t sh a ->
  t sh a ->
  t out_sh a
binaryCrossEntropy reduce probabilities targets = reduce $ lift2 forward backward probabilities targets
  where
    forward :: Tensor sh a -> Tensor sh a -> Tensor sh a
    forward p y = negate (y * E.max (log p) (-100) + (1 - y) * E.max (log (1 - p)) (-100))
    backward :: Tensor sh a -> Tensor sh a -> Tensor sh a -> Tensor sh a -> (Tensor sh a, Tensor sh a)
    backward p y _ seed =
      ( seed * (p - y) / E.max (p * (1 - p)) 1e-12,
        seed * (log (1 - p) - log p)
      )

-- | Binary cross entropy from logits, equivalent to applying sigmoid followed by 'binaryCrossEntropy' away from its numerical limits.
-- Computes @max x 0 - x * y + log (1 + exp (-abs x))@ without overflowing exponentials for finite logits.
-- Targets must have the same shape and contain probabilities in @[0, 1]@.
-- Suitable for both binary classification and independent multi-label predictions.
-- Keep the model's final output unnormalized and apply sigmoid separately when probabilities are needed.
binaryCrossEntropyWithLogits ::
  forall {t :: TensorLike} in_sh out_sh a.
  (Floating (t in_sh a), E.Ord (t in_sh a), E.Conditional (t in_sh a), E.Condition (t in_sh a) ~ E.Comparison (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
binaryCrossEntropyWithLogits reduce logits targets =
  reduce $ E.max logits 0 - logits * targets + softplus (negate (abs logits))

-- | Negative log likelihood @-sum (targets * logProbabilities)@ along the supplied class axis, followed by the caller's reduction.
-- Inputs have the same shape; targets are one-hot labels or class probabilities summing to one along the class axis.
-- The class axis must be nonempty and is removed before applying the reduction.
-- For probabilities p, pass @log p@; prefer 'crossEntropy' on logits to avoid underflow before taking logarithms.
negativeLogLikelihood ::
  forall {t :: TensorLike} axis sh out_sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    Num (TensorElement a),
    Num (t sh a)
  ) =>
  (t (ReduceShape '[axis] sh) a -> t out_sh a) ->
  t sh a ->
  t sh a ->
  t out_sh a
negativeLogLikelihood reduce logProbabilities targets = reduce $ R.sumAxes @'[axis] $ negate $ targets * logProbabilities

-- | Multiclass cross entropy from unnormalized logits and one-hot or probability targets.
-- Computes @-sum (targets * logSoftmax logits)@ along the supplied class axis, then applies the reduction.
-- Targets have the same shape as logits and must form probability distributions along the class axis.
-- No sigmoid or softmax should be applied to model outputs before this loss.
-- The class axis must be nonempty; all remaining axes describe independent observations.
-- For shape @'[batch, classes]@ use @crossEntropy \@1 T.mean logits targets@.
crossEntropy ::
  forall {t :: TensorLike} axis sh out_sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    Floating a,
    Floating (TensorElement a),
    E.Ord (TensorElement a),
    Num (t sh a)
  ) =>
  (t (ReduceShape '[axis] sh) a -> t out_sh a) ->
  t sh a ->
  t sh a ->
  t out_sh a
crossEntropy reduce logits = negativeLogLikelihood @axis reduce (R.logSoftmax @axis logits)

-- | Negative log likelihood from log probabilities and unchecked integer class labels.
-- Targets have the input shape with the class axis removed and are not differentiated.
-- Each label must lie in @[0, classes)@; invalid labels may cause invalid memory access.
-- No one-hot target tensor is constructed.
unsafeNegativeLogLikelihood ::
  forall {t :: TensorLike} axis sh out_sh a.
  (TensorOps t a, KnownNat axis, Shape sh, Shape (RemoveAxis axis sh), CanReduceNonEmpty '[axis] sh, Num (t (RemoveAxis axis sh) a)) =>
  (t (RemoveAxis axis sh) a -> t out_sh a) ->
  t sh a ->
  Tensor (RemoveAxis axis sh) Int ->
  t out_sh a
unsafeNegativeLogLikelihood reduce logProbabilities targets =
  reduce (negate (lift1 (\input -> Raw.unsafeGatherReduced @axis input targets) (Pullback.gatherReduced @axis targets) logProbabilities))

-- | Multiclass cross entropy from logits and unchecked integer class labels.
-- For logits @'[batch, classes]@, targets have shape @'[batch]@ and the class axis is @1@.
-- Each label must lie in @[0, classes)@; invalid labels may cause invalid memory access.
-- Targets carry no differentiation history; the loss propagates gradients through logits.
-- Use 'crossEntropy' for one-hot labels or probability distributions instead.
unsafeCrossEntropy ::
  forall {t :: TensorLike} axis sh out_sh a.
  (TensorOps t a, KnownNat axis, Shape sh, Shape (RemoveAxis axis sh), Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh, Floating a, Floating (TensorElement a), E.Ord (TensorElement a), Num (t (RemoveAxis axis sh) a)) =>
  (t (RemoveAxis axis sh) a -> t out_sh a) ->
  t sh a ->
  Tensor (RemoveAxis axis sh) Int ->
  t out_sh a
unsafeCrossEntropy reduce logits = unsafeNegativeLogLikelihood @axis reduce (R.logSoftmax @axis logits)

-- | Cross entropy with label smoothing and unchecked integer class labels.
-- Smoothing mixes the selected class with a uniform distribution without constructing one-hot targets.
-- A smoothing value of zero gives 'unsafeCrossEntropy', and one uses only the uniform distribution.
-- Smoothing must lie in @[0, 1]@; every target must lie in @[0, classes)@.
-- Invalid targets may cause invalid memory access; use 'smoothLabels' and 'crossEntropy' for probability targets.
unsafeCrossEntropySmoothed ::
  forall {t :: TensorLike} axis sh out_sh a.
  ( TensorOps t a,
    KnownNat axis,
    Shape sh,
    Shape (RemoveAxis axis sh),
    RemoveAxis axis sh ~ ReduceShape '[axis] sh,
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Floating (TensorElement a),
    E.Ord (TensorElement a),
    Fractional (t (RemoveAxis axis sh) a)
  ) =>
  (t (RemoveAxis axis sh) a -> t out_sh a) ->
  a ->
  t sh a ->
  Tensor (RemoveAxis axis sh) Int ->
  t out_sh a
unsafeCrossEntropySmoothed reduce smoothing logits targets
  | smoothing == 0 = reduce selected
  | smoothing == 1 = reduce uniform
  | otherwise = reduce $ realToFrac (1 - smoothing) * selected + realToFrac smoothing * uniform
  where
    logProbabilities = R.logSoftmax @axis logits
    selected = unsafeNegativeLogLikelihood @axis id logProbabilities targets
    uniform = negate (R.sumAxes @'[axis] logProbabilities) / fromIntegral (classCount @axis @sh)
