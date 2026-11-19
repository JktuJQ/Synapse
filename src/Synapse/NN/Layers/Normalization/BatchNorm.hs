{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoDeepSubsumption #-}

-- | Per-channel normalization with explicit running statistics.
module Synapse.NN.Layers.Normalization.BatchNorm (BatchNorm (..), BatchNormState (..), initialBatchNormState, batchNorm) where

import Control.Monad.Trans.State.Strict (State, state)
import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat)
import Synapse.NN.Forward (ForwardMode (..))
import Synapse.Tensor.Ops.Constraints (FloatingDType, NumDType)
import Synapse.Tensor.Ops.Definition (TensorOps (..))
import Synapse.Tensor.Ops.Math ()
import qualified Synapse.Tensor.Ops.Raw.Reduction as Raw
import qualified Synapse.Tensor.Ops.Reduction as R
import qualified Synapse.Tensor.Ops.Shape as T
import Synapse.Tensor.Representation (TensorLike)
import qualified Synapse.Tensor.Representation as Tensor
import Synapse.Tensor.Shapes.Axes (Axes, AxisSize, CanReduce, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Synapse.Tensor.Shapes.Normalization (BatchNormAxes)
import Synapse.Tensor.Structure (TensorStructure, WithTensor)

-- | Per-channel scale and bias for batch normalization.
data BatchNorm (channels :: Nat) (t :: TensorLike) a = BatchNorm
    { batchNormWeights :: t '[channels] a
    , batchNormBias :: t '[channels] a
    }
    deriving (Generic)

instance TensorStructure (BatchNorm channels Tensor.Tensor a) where
    type WithTensor t (BatchNorm channels Tensor.Tensor a) = BatchNorm channels t a

-- | Running per-channel statistics, independent of the trainable parameters.
data BatchNormState (channels :: Nat) (t :: TensorLike) a = BatchNormState
    { batchNormMean :: t '[channels] a
    , batchNormVariance :: t '[channels] a
    }
    deriving (Generic)

instance TensorStructure (BatchNormState channels Tensor.Tensor a) where
    type WithTensor t (BatchNormState channels Tensor.Tensor a) = BatchNormState channels t a

-- | Initialize running means to zero and running variances to one.
initialBatchNormState
    :: (KnownNat channels, NumDType a)
    => BatchNormState channels Tensor.Tensor a
initialBatchNormState = BatchNormState 0 1

{- | Normalize each channel over all other axes and apply its scale and bias.

The visible type argument selects the channel axis, such as @batchNorm \@1@ for NCHW data.
Arguments after the mode are epsilon, momentum, parameters and input.
Running statistics are carried by the returned state action.
Training updates running statistics as @(1 - momentum) * old + momentum * batch@.
The output uses population variance, while the running variance uses the unbiased estimate.
Evaluation uses the running statistics and preserves the state without inspecting momentum.
Empty training inputs preserve running statistics; a nonempty training region requires at least two elements per channel.
Epsilon must be positive and momentum must lie in @[0, 1]@.
-}
batchNorm
    :: forall axis channels sh t a xs
     . ( xs ~ BatchNormAxes axis sh
       , AxisSize axis sh ~ channels
       , ReduceShape xs sh ~ '[channels]
       , KnownNat channels
       , TensorOps t a
       , RealFloat a
       , FloatingDType a
       , Floating (t sh a)
       , Fractional (t '[channels] a)
       , Shape sh
       , Axes xs
       , CanReduce xs sh
       )
    => ForwardMode
    -> Double
    -> Double
    -> BatchNorm channels t a
    -> t sh a
    -> State (BatchNormState channels Tensor.Tensor a) (t sh a)
batchNorm mode epsilon momentum parameters input = state $ \running ->
    let apply mean variance =
            let normalized = (input - T.expandReduced @xs input mean) / sqrt (T.expandReduced @xs input variance + realToFrac epsilon)
             in normalized * T.expandReduced @xs input (batchNormWeights parameters)
                    + T.expandReduced @xs input (batchNormBias parameters)
        evaluate = (apply (fromTensor (batchNormMean running)) (fromTensor (batchNormVariance running)), running)
     in case mode of
            Eval -> evaluate
            Train
                | shapeSize @sh == 0 -> evaluate
                | count <= 1 -> error "batchNorm: training requires at least two values per channel"
                | otherwise ->
                    let mean = R.meanAxes @xs input
                        variance = R.varianceAxes @xs 0 input
                        update old current = realToFrac (1 - momentum) * old + realToFrac momentum * current
                        unbiased = toTensor variance * realToFrac (fromIntegral count / fromIntegral (count - 1) :: Double)
                        nextState =
                            BatchNormState (update (batchNormMean running) (toTensor mean)) (update (batchNormVariance running) unbiased)
                     in (apply mean variance, nextState)
  where
    count = Raw.reductionSize @xs @sh
