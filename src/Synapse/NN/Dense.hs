{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Fully connected layers acting on the last input dimension.
module Synapse.NN.Dense (Dense (..)) where

import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat)
import Synapse.Autograd.Differentiable (Differentiable (..))
import qualified Synapse.Autograd.Tensor as AG
import Synapse.NN.Model (Model (..))
import Synapse.Tensor.Ops.Math (LinearAlgebra, matmul)
import Synapse.Tensor.Ops.Shape (ShapeOps, broadcast)
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Matmul (MatmulShape)
import Synapse.Tensor.Type (TensorLike)
import qualified Synapse.Tensor.Type as T

-- | Parameters of an affine transformation @input * weights + bias@.
-- Weights have shape @'[in_features, out_features]@ and bias has shape @'[out_features]@.
-- 'forward' preserves leading input dimensions and replaces the last dimension with @out_features@.
-- Construct parameters with any initialization functions before registering them for differentiation.
--
-- @
-- parameters :: Dense 2 3 T.Tensor Float
-- parameters = Dense (NN.xavierUniform \@'[2, 3] 2 3 1 generator) (T.zeros \@'[3])
-- @
data Dense (in_features :: Nat) (out_features :: Nat) (t :: TensorLike) a = Dense
  { -- | Weights mapping input features to output features.
    denseWeights :: t '[in_features, out_features] a,
    -- | Bias added to every output feature vector.
    denseBias :: t '[out_features] a
  }
  deriving (Generic)

instance
  (Num (T.Tensor '[in_features, out_features] a), Num (T.Tensor '[out_features] a)) =>
  Differentiable (Dense in_features out_features T.Tensor a)
  where
  type WithGrad s (Dense in_features out_features T.Tensor a) = Dense in_features out_features (AG.Tensor s) a

instance
  ( KnownNat in_features,
    KnownNat out_features,
    Shape in_sh,
    out_sh ~ MatmulShape in_sh '[in_features, out_features],
    Shape out_sh,
    CanBroadcast '[out_features] out_sh,
    LinearAlgebra t a,
    ShapeOps t a,
    Num (t out_sh a)
  ) =>
  Model (Dense in_features out_features) in_sh out_sh t a
  where
  forward parameters input = matmul input (denseWeights parameters) + broadcast (denseBias parameters)
