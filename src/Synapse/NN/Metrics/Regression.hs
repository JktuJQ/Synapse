{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

-- | Regression metrics for predictions and targets of matching shapes.
module Synapse.NN.Metrics.Regression (mse, mae, rmse, r2) where

import Synapse.NN.Loss.Regression (mae, mse)
import Synapse.Tensor.Ops.Constraints (FractionalDType)
import Synapse.Tensor.Ops.Definition (TensorOps)
import Synapse.Tensor.Ops.Shape (broadcast)
import Synapse.Tensor.Representation (TensorLike)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)

import qualified Synapse.Tensor.Ops.Reduction as R

{- | Compute the square root of reduced squared differences.

Use @rmse T.mean predictions targets@ for root mean squared error.
The reduction determines the output shape and the treatment of empty inputs, as in 'mse'.
-}
rmse
    :: forall {t :: TensorLike} in_sh out_sh a
     . (Num (t in_sh a), Floating (t out_sh a))
    => (t in_sh a -> t out_sh a)
    -> t in_sh a
    -> t in_sh a
    -> t out_sh a
rmse reduce predictions targets = sqrt $ mse reduce predictions targets

{- | Compute the coefficient of determination
@1 - sum ((predictions - targets)^2) / sum ((targets - mean targets)^2)@.

All elements are treated as observations with a single global target mean, returning a scalar score.
A score of one indicates perfect predictions; zero matches predicting the target mean and negative scores indicate a worse fit.
At least two elements are required; smaller inputs raise an error.
A zero denominator gives NaN for perfect predictions and negative infinity otherwise.
-}
r2
    :: forall {t :: TensorLike} sh a
     . (Shape sh, TensorOps t a, RealFloat a, FractionalDType a, Num (t sh a), Fractional (t '[] a))
    => t sh a
    -> t sh a
    -> t '[] a
r2 predictions targets
    | shapeSize @sh < 2 = error "Synapse.NN.r2: at least two observations are required"
    | otherwise =
        let centered = targets - broadcast (R.mean targets)
            residual = mse R.sum predictions targets
            total = R.sum $ centered * centered
         in 1 - residual / total
