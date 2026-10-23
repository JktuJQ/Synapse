{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Regression metrics for equally shaped predictions and targets.
-- 'mse' and 'mae' are shared with "Synapse.NN.Loss".
-- Ordinary and differentiable tensors use the same functions.
-- Percentage errors are returned as fractions; multiply by 100 to express percentages.
-- Metrics without a reduction argument return a scalar using all elements as observations.
module Synapse.NN.Metrics
  ( mse,
    mae,
    rmse,
    r2,
    meanError,
    maxError,
    mape,
    smape,
    wmape,
    msle,
    rmsle,
    rae,
    rse,
    explainedVariance,
    pearson,
  )
where

import Synapse.NN.Loss (mae, mse)
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Reduction as R
import Synapse.Tensor.Ops.Shape (ShapeOps, broadcast)
import Synapse.Tensor.Shapes.Axes (NonEmpty)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Synapse.Tensor.Type (TensorLike)

center ::
  (Shape sh, R.MeanReductions t a, ShapeOps t a, Num (t sh a)) =>
  t sh a ->
  t sh a
center input = input - broadcast (R.mean input)

checkedEpsilon :: (RealFloat a, Fractional b) => a -> b
checkedEpsilon epsilon
  | isNaN epsilon || isInfinite epsilon || epsilon <= 0 = error "Synapse.NN.Metrics: epsilon must be finite and positive"
  | otherwise = realToFrac epsilon

atLeastTwo :: forall sh b. (Shape sh) => String -> b -> b
atLeastTwo name result
  | shapeSize @sh < 2 = error $ "Synapse.NN." ++ name ++ ": at least two observations are required"
  | otherwise = result

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
  (Shape sh, R.Reductions t a, R.MeanReductions t a, ShapeOps t a, Num (t sh a), Fractional (t '[] a)) =>
  t sh a ->
  t sh a ->
  t '[] a
r2 predictions targets =
  atLeastTwo @sh "r2" $
    let centered = center targets
        residual = mse R.sum predictions targets
        total = R.sum (centered * centered)
     in 1 - residual / total

-- | Reduce signed errors @predictions - targets@.
-- Use @meanError T.mean predictions targets@ to measure prediction bias.
meanError ::
  forall {t :: TensorLike} in_sh out_sh a.
  (Num (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
meanError reduce predictions targets = reduce $ predictions - targets

-- | Compute the largest absolute error across all elements.
-- Empty tensors are rejected at compile time and NaN values propagate to the result.
maxError ::
  forall {t :: TensorLike} sh a.
  (Shape sh, NonEmpty sh, R.OrderedReductions t a, Num (t sh a)) =>
  t sh a ->
  t sh a ->
  t '[] a
maxError predictions targets = R.maximum $ abs (predictions - targets)

-- | Reduce absolute percentage errors @abs (predictions - targets) / max epsilon (abs targets)@.
-- Arguments are the reduction, a finite positive epsilon, predictions, then targets.
-- Use a mean reduction for MAPE; epsilon limits the effect of zero or near-zero targets.
mape ::
  forall {t :: TensorLike} in_sh out_sh a.
  (RealFloat a, Fractional (t in_sh a), E.Ord (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  a ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
mape reduce epsilon predictions targets =
  let floorValue = checkedEpsilon epsilon
   in floorValue `seq` reduce (abs (predictions - targets) / E.max floorValue (abs targets))

-- | Reduce symmetric absolute percentage errors @2 * abs (predictions - targets) / max epsilon (abs predictions + abs targets)@.
-- Arguments and epsilon requirements are the same as for 'mape'.
-- A mean reduction gives SMAPE between zero and two for finite inputs.
smape ::
  forall {t :: TensorLike} in_sh out_sh a.
  (RealFloat a, Fractional (t in_sh a), E.Ord (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  a ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
smape reduce epsilon predictions targets =
  let floorValue = checkedEpsilon epsilon
   in floorValue `seq` reduce (2 * (abs (predictions - targets) / E.max floorValue (abs predictions + abs targets)))

-- | Compute weighted MAPE @sum (abs (predictions - targets)) / max epsilon (sum (abs targets))@.
-- The first argument is a finite positive epsilon; empty inputs produce zero.
-- Errors are weighted by target magnitude and the score can exceed one.
wmape ::
  forall {t :: TensorLike} sh a.
  (Shape sh, RealFloat a, R.Reductions t a, Num (t sh a), Fractional (t '[] a), E.Ord (t '[] a)) =>
  a ->
  t sh a ->
  t sh a ->
  t '[] a
wmape epsilon predictions targets =
  let floorValue = checkedEpsilon epsilon
   in floorValue `seq` (mae R.sum predictions targets / E.max floorValue (R.sum (abs targets)))

-- | Reduce squared logarithmic errors @(log (1 + predictions) - log (1 + targets))²@.
-- Use a mean reduction for MSLE.
-- Inputs must be greater than minus one; values outside the logarithm's domain produce nonfinite results.
msle ::
  forall {t :: TensorLike} in_sh out_sh a.
  (Floating (t in_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
msle reduce predictions targets = mse reduce (log (1 + predictions)) (log (1 + targets))

-- | Compute the square root of reduced squared logarithmic errors.
-- Use @rmsle T.mean predictions targets@ for RMSLE; the input domain is the same as for 'msle'.
rmsle ::
  forall {t :: TensorLike} in_sh out_sh a.
  (Floating (t in_sh a), Floating (t out_sh a)) =>
  (t in_sh a -> t out_sh a) ->
  t in_sh a ->
  t in_sh a ->
  t out_sh a
rmsle reduce predictions targets = sqrt $ msle reduce predictions targets

-- | Compute relative absolute error @sum (abs (predictions - targets)) / sum (abs (targets - mean targets))@.
-- At least two observations are required; smaller inputs raise an error.
-- A zero denominator gives NaN for perfect predictions and positive infinity otherwise.
rae ::
  forall {t :: TensorLike} sh a.
  (Shape sh, R.Reductions t a, R.MeanReductions t a, ShapeOps t a, Num (t sh a), Fractional (t '[] a)) =>
  t sh a ->
  t sh a ->
  t '[] a
rae predictions targets =
  atLeastTwo @sh "rae" $
    mae R.sum predictions targets / R.sum (abs (center targets))

-- | Compute relative squared error @sum ((predictions - targets)²) / sum ((targets - mean targets)²)@.
-- At least two observations are required; smaller inputs raise an error.
-- A zero denominator gives NaN for perfect predictions and positive infinity otherwise.
rse ::
  forall {t :: TensorLike} sh a.
  (Shape sh, R.Reductions t a, R.MeanReductions t a, ShapeOps t a, Num (t sh a), Fractional (t '[] a)) =>
  t sh a ->
  t sh a ->
  t '[] a
rse predictions targets =
  atLeastTwo @sh "rse" $
    let centered = center targets
     in mse R.sum predictions targets / R.sum (centered * centered)

-- | Compute explained variance @1 - variance (predictions - targets) / variance targets@.
-- Unlike 'r2', this score does not penalize a constant prediction offset.
-- At least two observations are required; a zero target variance produces NaN or negative infinity.
explainedVariance ::
  forall {t :: TensorLike} sh a.
  (Shape sh, R.Reductions t a, R.MeanReductions t a, ShapeOps t a, Num (t sh a), Fractional (t '[] a)) =>
  t sh a ->
  t sh a ->
  t '[] a
explainedVariance predictions targets =
  atLeastTwo @sh "explainedVariance" $
    let errors = center (predictions - targets)
        centered = center targets
     in 1 - R.sum (errors * errors) / R.sum (centered * centered)

-- | Compute Pearson correlation @sum (p * t) / (sqrt (sum (p²)) * sqrt (sum (t²)))@, where p and t are centered predictions and targets.
-- Positive and negative one indicate perfect positive and negative linear relationships respectively.
-- Empty tensors are rejected at compile time and singleton inputs raise an error.
-- Constant predictions or targets produce NaN.
pearson ::
  forall {t :: TensorLike} sh a.
  ( Shape sh,
    NonEmpty sh,
    R.Reductions t a,
    R.MeanReductions t a,
    R.OrderedReductions t a,
    ShapeOps t a,
    Num (t sh a),
    Floating (t '[] a),
    E.Eq (t '[] a),
    E.Conditional (t '[] a),
    E.Condition (t '[] a) ~ E.Comparison (t '[] a)
  ) =>
  t sh a ->
  t sh a ->
  t '[] a
pearson predictions targets =
  atLeastTwo @sh "pearson" $
    let p = center predictions
        t = center targets
        score = R.sum (p * t) / (sqrt (R.sum (p * p)) * sqrt (R.sum (t * t)))
        constantPredictions = R.minimum predictions E.== R.maximum predictions
        constantTargets = R.minimum targets E.== R.maximum targets
     in E.select constantPredictions (0 / 0) (E.select constantTargets (0 / 0) score)
