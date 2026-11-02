{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Differentiable normalization statistics and their expansion to the input shape.
module Synapse.NN.Layers.Normalization.Statistics (expand, moments) where

import qualified Synapse.Autograd.Tensor.Ops.Primitives.Shape as S
import Synapse.Tensor.Ops (TensorOps (..))
import Synapse.Tensor.Ops.Primitives.Math ()
import qualified Synapse.Tensor.Ops.Primitives.Reduction as Raw
import qualified Synapse.Tensor.Ops.Reduction as R
import Synapse.Tensor.Shapes.Axes (Axes (..), CanReduce, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (TensorElement)

-- | Expand reduced statistics back across the axes from which they were computed.
expand :: forall xs sh t a. (TensorOps t a, Shape sh, Shape (ReduceShape xs sh), Axes xs, Num (TensorElement a)) => t sh a -> t (ReduceShape xs sh) a -> t sh a
expand reference = lift1 (S.expandReduced (axes @xs) (primal reference)) (\_ _ seed -> Raw.sumAxes @xs seed)

-- | Compute the mean and population variance independently in each reduction region.
moments :: forall xs sh t a. (TensorOps t a, RealFloat a, Floating (TensorElement a), Floating (t sh a), Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) => t sh a -> (t (ReduceShape xs sh) a, t (ReduceShape xs sh) a)
moments input =
  let mean = R.meanAxes @xs input
      centered = input - expand @xs input mean
   in (mean, R.meanAxes @xs (centered * centered))
