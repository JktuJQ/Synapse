{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Per-channel normalization with explicit running statistics.
module Synapse.NN.Layers.Normalization.BatchNorm (BatchNorm (..), BatchNormState (..), initialBatchNormState, batchNorm) where

import Control.Monad.Trans.State.Strict (State, state)
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat)
import Synapse.NN.Forward (ForwardMode (..))
import Synapse.NN.Layers.Normalization.Statistics (expand, moments)
import Synapse.Tensor.Ops (TensorOps (..))
import Synapse.Tensor.Ops.Primitives.Math ()
import qualified Synapse.Tensor.Ops.Primitives.Reduction as Raw
import Synapse.Tensor.Shapes.Axes (Axes, AxisSize, CanReduce, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Synapse.Tensor.Shapes.Normalization (BatchNormAxes)
import Synapse.Tensor.Type (TensorElement, TensorLike, WithTensor)
import qualified Synapse.Tensor.Type as Tensor

-- | Per-channel scale and bias for batch normalization.
-- Initialize weights to ones and bias to zeros.
data BatchNorm (channels :: Nat) (t :: TensorLike) a = BatchNorm
  { -- | Scale shared across all positions in a channel.
    batchNormWeights :: t '[channels] a,
    -- | Offset shared across all positions in a channel.
    batchNormBias :: t '[channels] a
  }
  deriving (Generic)

type instance WithTensor t (BatchNorm channels Tensor.Tensor a) = BatchNorm channels t a

-- | Running per-channel statistics, independent of the trainable parameters.
data BatchNormState (channels :: Nat) (t :: TensorLike) a = BatchNormState
  { -- | Exponentially averaged per-channel means.
    batchNormMean :: t '[channels] a,
    -- | Exponentially averaged unbiased per-channel variances.
    batchNormVariance :: t '[channels] a
  }
  deriving (Generic)

type instance WithTensor t (BatchNormState channels Tensor.Tensor a) = BatchNormState channels t a

-- | Initialize running means to zero and running variances to one.
initialBatchNormState :: (KnownNat channels, Num a, Num (TensorElement a), Tensor.DType a) => BatchNormState channels Tensor.Tensor a
initialBatchNormState = BatchNormState 0 1

-- | Normalize each channel over all other axes and apply its scale and bias.
-- The visible type argument selects the channel axis, such as @batchNorm \@1@ for NCHW data.
-- Arguments after the mode are epsilon, momentum, parameters and input.
-- Running statistics are carried by the returned state action.
-- Training updates running statistics as @(1 - momentum) * old + momentum * batch@.
-- The output uses population variance, while the running variance uses the unbiased estimate.
-- Evaluation uses the running statistics and preserves the state without inspecting momentum.
-- Empty training inputs preserve running statistics; a nonempty training region requires at least two elements per channel.
-- Epsilon must be positive and momentum must lie in @[0, 1]@.
batchNorm ::
  forall axis channels sh t a xs.
  (xs ~ BatchNormAxes axis sh, AxisSize axis sh ~ channels, ReduceShape xs sh ~ '[channels], KnownNat channels, TensorOps t a, RealFloat a, Floating (TensorElement a), Floating (t sh a), Shape sh, Axes xs, CanReduce xs sh) =>
  ForwardMode ->
  Double ->
  Double ->
  BatchNorm channels t a ->
  t sh a ->
  State (BatchNormState channels Tensor.Tensor a) (t sh a)
batchNorm mode epsilon momentum parameters input = state $ \running ->
  let apply mean variance =
        let normalized = (input - expand @xs input mean) / sqrt (expand @xs input variance + realToFrac epsilon)
         in normalized * expand @xs input (batchNormWeights parameters) + expand @xs input (batchNormBias parameters)
      evaluate = (apply (literal (batchNormMean running)) (literal (batchNormVariance running)), running)
   in case mode of
        Eval -> evaluate
        Train
          | shapeSize @sh == 0 -> evaluate
          | count <= 1 -> error "batchNorm: training requires at least two values per channel"
          | otherwise ->
              let (mean, variance) = moments @xs input
                  update old current = realToFrac (1 - momentum) * old + realToFrac momentum * current
                  unbiased = primal variance * realToFrac (fromIntegral count / fromIntegral (count - 1) :: Double)
                  nextState = BatchNormState (update (batchNormMean running) (primal mean)) (update (batchNormVariance running) unbiased)
               in (apply mean variance, nextState)
  where
    count = Raw.reductionSize @xs @sh
