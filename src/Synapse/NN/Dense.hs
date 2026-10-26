{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Fully connected layers acting on the last input dimension.
module Synapse.NN.Dense (Dense (..), dense) where

import GHC.Generics (Generic)
import GHC.TypeLits (KnownNat, Nat)
import Synapse.Tensor.Ops (TensorOps)
import Synapse.Tensor.Ops.Math (matmul)
import Synapse.Tensor.Ops.Shape (broadcast)
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Matmul (MatmulShape)
import Synapse.Tensor.Type (TensorLike, WithTensor)
import qualified Synapse.Tensor.Type as T

-- | Parameters of an affine transformation @input * weights + bias@.
-- Weights have shape @'[in_features, out_features]@ and bias has shape @'[out_features]@.
-- 'dense' preserves leading input dimensions and replaces the last dimension with @out_features@.
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

type instance WithTensor t (Dense in_features out_features T.Tensor a) = Dense in_features out_features t a

-- | Apply @input * weights + bias@ along the last input dimension.
-- The input must have at least one dimension, with @in_features@ elements on the last axis.
-- Leading input dimensions are preserved, and the last dimension becomes @out_features@.
dense ::
  ( KnownNat in_features,
    KnownNat out_features,
    Shape in_sh,
    out_sh ~ MatmulShape in_sh '[in_features, out_features],
    Shape out_sh,
    CanBroadcast '[out_features] out_sh,
    TensorOps t a,
    Num (T.TensorElement a),
    Num (t out_sh a)
  ) =>
  Dense in_features out_features t a ->
  t in_sh a ->
  t out_sh a
dense parameters input = matmul input (denseWeights parameters) + broadcast (denseBias parameters)
