{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Regression and classification metrics.
-- Regression predictions and targets have matching shapes; classification metrics document their label conventions.
-- Ordinary and differentiable tensors use the same functions.
-- Classification scores have no differentiation history and pool all observation dimensions.
-- Per-class functions and confusion matrices retain their documented class dimensions.
module Synapse.NN.Metrics
  ( mse,
    mae,
    rmse,
    r2,
    binaryAccuracy,
    accuracy,
    Averaging (..),
    precision,
    recall,
    f1,
    precisionPerClass,
    recallPerClass,
    f1PerClass,
    confusionMatrix,
    binaryPrecision,
    binaryRecall,
    binaryF1,
  )
where

import qualified Data.Array.Accelerate as A
import Synapse.NN.Loss (mae, mse)
import Synapse.Tensor.Ops (TensorOps (primal))
import qualified Synapse.Tensor.Ops.Construction as C
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Reduction as R
import Synapse.Tensor.Ops.Shape (broadcast)
import Synapse.Tensor.Shapes.Axes (Axes, AxisSize, CanReduceNonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..), TensorLike)

-- | Combine class-wise precision, recall, or F1 statistics.
-- Every class on the prediction axis is included, even when absent from the observations.
data Averaging
  = -- | Sum counts before computing the score.
    Micro
  | -- | Average the individual class scores with equal weight.
    Macro
  | -- | Average class scores weighted by the number of target observations in each class.
    Weighted
  deriving (Eq, Show)

-- | Divide counts, returning zero when the denominator is zero.
ratio :: (Shape sh, DType a, Fractional a, Fractional (TensorElement a), E.Ord (TensorElement a)) => Tensor sh a -> Tensor sh a -> Tensor sh a
ratio numerator denominator = E.select (denominator E.== 0) 0 $ numerator / E.select (denominator E.== 0) 1 denominator

meanMatches :: forall sh a. (Shape sh, DType a, RealFloat a, Fractional (TensorElement a)) => Tensor sh Bool -> Tensor '[] a
meanMatches matches = R.mean (E.select matches (C.full @sh 1) (C.full @sh 0))

-- | Compute the square root of reduced squared differences.
-- Use @rmse T.mean predictions targets@ for root mean squared error.
-- The reduction determines the output shape and the treatment of empty inputs, as in 'mse'.
rmse ::
  forall {t :: TensorLike} in_sh out_sh a.
  (Num (t in_sh a), Floating (t out_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
rmse reduce predictions targets = sqrt $ mse reduce predictions targets

-- | Compute the coefficient of determination @1 - sum ((predictions - targets)²) / sum ((targets - mean targets)²)@.
-- All elements are treated as observations with a single global target mean, returning a scalar score.
-- A score of one indicates perfect predictions; zero matches predicting the target mean and negative scores indicate a worse fit.
-- At least two elements are required; smaller inputs raise an error.
-- A zero denominator gives NaN for perfect predictions and negative infinity otherwise.
r2 ::
  forall {t :: TensorLike} sh a.
  (Shape sh, TensorOps t a, RealFloat a, Fractional (TensorElement a), Num (t sh a), Fractional (t '[] a)) =>
  t sh a ->
  t sh a ->
  t '[] a
r2 predictions targets
  | shapeSize @sh < 2 = error "Synapse.NN.r2: at least two observations are required"
  | otherwise =
      let centered = targets - broadcast (R.mean targets)
          residual = mse R.sum predictions targets
          total = R.sum $ centered * centered
       in 1 - residual / total

-- | Fraction of correct binary or independent multi-label predictions.
-- Arguments are a decision threshold, predictions, and equally shaped targets.
-- Use threshold 0 for logits or 0.5 for probabilities; a prediction equal to the threshold is positive.
-- Targets at least 0.5 are positive, including ordinary zero/one labels.
-- All elements count independently; empty inputs return NaN.
-- The result is an ordinary scalar tensor with no gradient history.
binaryAccuracy ::
  forall {t :: TensorLike} sh a.
  (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a)) =>
  a ->
  t sh a ->
  t sh a ->
  Tensor '[] a
binaryAccuracy threshold predictions targets =
  let predicted = primal predictions E.>= C.full @sh threshold
      actual = primal targets E.>= 0.5
   in meanMatches @sh @a (predicted E.== actual)

-- | Fraction of correct multiclass predictions along the supplied class axis.
-- Predictions may be logits or probabilities; the greatest score determines the predicted class.
-- Targets contain zero-based integer class indices and have the prediction shape with the class axis removed.
-- Ties select the first class, as in argmax; empty observation sets return NaN.
-- The result is an ordinary scalar tensor with no gradient history.
accuracy ::
  forall {t :: TensorLike} axis sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Fractional (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  Tensor '[] a
accuracy predictions targets = meanMatches @(ReduceShape '[axis] sh) @a (R.argmax @axis predictions E.== targets)

-- | Count true positives, predicted positives, and target support without allocating a class-by-class matrix.
-- Invalid target labels are excluded from every count.
classificationCounts ::
  forall {t :: TensorLike} axis sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Shape '[AxisSize axis sh],
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Fractional (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  (Tensor '[AxisSize axis sh] a, Tensor '[AxisSize axis sh] a, Tensor '[AxisSize axis sh] a)
classificationCounts predictions (Tensor targets) =
  let Tensor predicted = R.argmax @axis predictions
      classes = shapeSize @'[AxisSize axis sh]
      zeros = A.fill (A.index1 $ A.constant classes) $ unTensorElement (0 :: TensorElement a)
      ones = A.fill (A.shape targets) $ unTensorElement (1 :: TensorElement a)
      add :: A.Exp a -> A.Exp a -> A.Exp a
      add x y = unTensorElement $ TensorElement x + TensorElement y
      count correct usePrediction =
        Tensor $
          A.permute
            add
            zeros
            ( \ix ->
                let actual = targets A.! ix
                    guessed = predicted A.! ix
                    valid = actual A.>= 0 A.&& actual A.< A.constant classes
                    selected = if usePrediction then guessed else actual
                    include = if correct then valid A.&& guessed A.== actual else valid
                 in A.cond include (A.Just_ $ A.index1 selected) A.Nothing_
            )
            ones
   in (count True False, count False True, count False False)

-- | Compute a score from pooled counts or aggregate its class-wise values.
averageCounts ::
  forall classes a.
  (Shape '[classes], DType a, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a)) =>
  (forall sh. (Shape sh) => Tensor sh a -> Tensor sh a -> Tensor sh a -> Tensor sh a) ->
  Averaging ->
  (Tensor '[classes] a, Tensor '[classes] a, Tensor '[classes] a) ->
  Tensor '[] a
averageCounts score averaging (tp, predicted, support) =
  case averaging of
    Micro -> score (R.sum tp) (R.sum predicted) $ R.sum support
    Macro -> ratio (R.sum $ score tp predicted support) $ C.full @'[] $ fromIntegral $ shapeSize @'[classes]
    Weighted -> ratio (R.sum $ score tp predicted support * support) $ R.sum support

-- | Precision @TP / (TP + FP)@ for each class in class-axis order.
-- Predictions are logits or probabilities; targets are zero-based class indices with the class axis removed.
-- All other dimensions are pooled, and argmax ties select the first class.
-- Out-of-range targets are ignored; a zero denominator gives zero, including for absent classes and empty observations.
-- The result is an ordinary tensor without gradient history.
precisionPerClass ::
  forall {t :: TensorLike} axis sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Shape '[AxisSize axis sh],
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Fractional (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  Tensor '[AxisSize axis sh] a
precisionPerClass predictions targets =
  let (tp, predicted, _) = classificationCounts @axis predictions targets
   in ratio tp predicted

-- | Precision @TP / (TP + FP)@, using the supplied class averaging strategy.
-- Label conventions and zero-denominator behavior match 'precisionPerClass'.
-- Micro pools counts, macro averages scores over all declared classes, and weighted uses target support.
-- For epoch-level scores, pool observations or counts rather than averaging batch scores.
precision ::
  forall {t :: TensorLike} axis sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Shape '[AxisSize axis sh],
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Fractional (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  Averaging ->
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  Tensor '[] a
precision averaging predictions targets =
  averageCounts (\tp predicted _ -> ratio tp predicted) averaging $ classificationCounts @axis predictions targets

-- | Recall @TP / (TP + FN)@ for each class in class-axis order.
-- Predictions are logits or probabilities; targets are zero-based class indices with the class axis removed.
-- All other dimensions are pooled, and argmax ties select the first class.
-- Out-of-range targets are ignored; a zero denominator gives zero, including for absent classes and empty observations.
-- The result is an ordinary tensor without gradient history.
recallPerClass ::
  forall {t :: TensorLike} axis sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Shape '[AxisSize axis sh],
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Fractional (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  Tensor '[AxisSize axis sh] a
recallPerClass predictions targets =
  let (tp, _, support) = classificationCounts @axis predictions targets
   in ratio tp support

-- | Recall @TP / (TP + FN)@, using the supplied class averaging strategy.
-- Label conventions and zero-denominator behavior match 'recallPerClass'.
-- Micro pools counts, macro averages scores over all declared classes, and weighted uses target support.
-- For epoch-level scores, pool observations or counts rather than averaging batch scores.
recall ::
  forall {t :: TensorLike} axis sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Shape '[AxisSize axis sh],
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Fractional (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  Averaging ->
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  Tensor '[] a
recall averaging predictions targets =
  averageCounts (\tp _ support -> ratio tp support) averaging $ classificationCounts @axis predictions targets

-- | F1 score @2 * TP / (2 * TP + FP + FN)@ for each class in class-axis order.
-- Predictions are logits or probabilities; targets are zero-based class indices with the class axis removed.
-- All other dimensions are pooled, and argmax ties select the first class.
-- Out-of-range targets are ignored; a zero denominator gives zero, including for absent classes and empty observations.
-- The result is an ordinary tensor without gradient history.
f1PerClass ::
  forall {t :: TensorLike} axis sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Shape '[AxisSize axis sh],
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Fractional (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  Tensor '[AxisSize axis sh] a
f1PerClass predictions targets =
  let (tp, predicted, support) = classificationCounts @axis predictions targets
   in ratio (2 * tp) $ predicted + support

-- | F1 score @2 * TP / (2 * TP + FP + FN)@, using the supplied class averaging strategy.
-- Label conventions and zero-denominator behavior match 'f1PerClass'.
-- Micro pools counts, macro averages scores over all declared classes, and weighted uses target support.
-- For epoch-level scores, pool observations or counts rather than averaging batch scores.
f1 ::
  forall {t :: TensorLike} axis sh a.
  ( TensorOps t a,
    Shape sh,
    Shape (ReduceShape '[axis] sh),
    Shape '[AxisSize axis sh],
    Axes '[axis],
    CanReduceNonEmpty '[axis] sh,
    RealFloat a,
    Fractional (TensorElement a),
    E.Ord (TensorElement a)
  ) =>
  Averaging ->
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  Tensor '[] a
f1 averaging predictions targets =
  averageCounts (\tp predicted support -> ratio (2 * tp) $ predicted + support) averaging $ classificationCounts @axis predictions targets

-- | Integer confusion matrix with target classes on rows and predicted classes on columns.
-- Predictions may be logits or probabilities; ties choose the first class.
-- Observation dimensions are pooled, out-of-range targets are ignored, and empty observations give a zero matrix.
-- Unlike precision, recall, and F1, this function allocates a class-by-class matrix.
confusionMatrix ::
  forall {t :: TensorLike} axis sh a.
  (TensorOps t a, Shape sh, Shape (ReduceShape '[axis] sh), Shape '[AxisSize axis sh], Axes '[axis], CanReduceNonEmpty '[axis] sh, E.Ord (TensorElement a)) =>
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int ->
  Tensor '[AxisSize axis sh, AxisSize axis sh] Int
confusionMatrix predictions (Tensor targets) =
  let Tensor predicted = R.argmax @axis predictions
      classes = shapeSize @'[AxisSize axis sh]
      zeros = A.fill (A.index2 (A.constant classes) (A.constant classes)) 0
      ones = A.fill (A.shape targets) 1
      destination ix =
        let actual = targets A.! ix
         in A.cond (actual A.>= 0 A.&& actual A.< A.constant classes) (A.Just_ $ A.index2 actual $ predicted A.! ix) A.Nothing_
   in Tensor $ A.permute (+) zeros destination ones

-- | Count positive-label true positives, predictions, and targets with an explicit threshold.
binaryCounts ::
  forall {t :: TensorLike} sh a.
  (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a)) =>
  a ->
  t sh a ->
  t sh a ->
  (Tensor '[] a, Tensor '[] a, Tensor '[] a)
binaryCounts threshold predictions targets =
  let predicted = primal predictions E.>= C.full @sh threshold
      actual = primal targets E.>= 0.5
      count mask = R.sum $ E.select mask (C.full @sh 1) $ C.full @sh 0
   in (count $ predicted E.&& actual, count predicted, count actual)

-- | Precision @TP / (TP + FP)@ for the positive binary label.
-- Arguments are a threshold, predictions, and targets of the same shape.
-- Use threshold 0 for logits or 0.5 for probabilities; values equal to the threshold are positive.
-- Targets at least 0.5 are positive, and all elements are pooled for independent multi-label predictions.
-- Zero denominators and empty inputs give zero; the result has no gradient history.
binaryPrecision ::
  forall {t :: TensorLike} sh a.
  (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a)) =>
  a ->
  t sh a ->
  t sh a ->
  Tensor '[] a
binaryPrecision threshold predictions targets =
  let (tp, predicted, _) = binaryCounts threshold predictions targets
   in ratio tp predicted

-- | Recall @TP / (TP + FN)@ for the positive binary label.
-- Arguments are a threshold, predictions, and targets of the same shape.
-- Use threshold 0 for logits or 0.5 for probabilities; values equal to the threshold are positive.
-- Targets at least 0.5 are positive, and all elements are pooled for independent multi-label predictions.
-- Zero denominators and empty inputs give zero; the result has no gradient history.
binaryRecall ::
  forall {t :: TensorLike} sh a.
  (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a)) =>
  a ->
  t sh a ->
  t sh a ->
  Tensor '[] a
binaryRecall threshold predictions targets =
  let (tp, _, support) = binaryCounts threshold predictions targets
   in ratio tp support

-- | F1 score @2 * TP / (2 * TP + FP + FN)@ for the positive binary label.
-- Arguments are a threshold, predictions, and targets of the same shape.
-- Use threshold 0 for logits or 0.5 for probabilities; values equal to the threshold are positive.
-- Targets at least 0.5 are positive, and all elements are pooled for independent multi-label predictions.
-- Zero denominators and empty inputs give zero; the result has no gradient history.
binaryF1 ::
  forall {t :: TensorLike} sh a.
  (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a)) =>
  a ->
  t sh a ->
  t sh a ->
  Tensor '[] a
binaryF1 threshold predictions targets =
  let (tp, predicted, support) = binaryCounts threshold predictions targets
   in ratio (2 * tp) $ predicted + support
