{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Differentiable normalization statistics and their expansion to the input shape.
module Synapse.NN.Layers.Normalization.Statistics (expand, moments) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import qualified Synapse.Autograd.Tensor.Ops.Pullback.Shape as S
import Synapse.Tensor.Ops.Definition (TensorOps (..), withVJP)
import Synapse.Tensor.Ops.Math ()
import qualified Synapse.Tensor.Ops.Raw.Reduction as Raw
import qualified Synapse.Tensor.Ops.Reduction as R
import Synapse.Tensor.Representation (TensorElement)
import Synapse.Tensor.Shapes.Axes (Axes (..), CanReduce, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape)

-- | Expand reduced statistics back across the axes from which they were computed.
expand
    :: forall xs sh t a
     . (TensorOps t a, Shape sh, Shape (ReduceShape xs sh), Axes xs, Num (TensorElement a))
    => t sh a
    -> t (ReduceShape xs sh) a
    -> t sh a
expand reference =
    withVJP
        (Proxy @(() :: Constraint))
        (S.expandReduced (axes @xs) (toTensor reference))
        (\_ _ seed -> Raw.sumAxes @xs seed)

-- | Compute the mean and population variance independently in each reduction region.
moments
    :: forall xs sh t a
     . ( TensorOps t a
       , RealFloat a
       , Floating (TensorElement a)
       , Floating (t sh a)
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       )
    => t sh a
    -> (t (ReduceShape xs sh) a, t (ReduceShape xs sh) a)
moments input =
    let mean = R.meanAxes @xs input
        centered = input - expand @xs input mean
     in (mean, R.meanAxes @xs (centered * centered))
