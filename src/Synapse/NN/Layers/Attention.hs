{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | Scaled dot-product and multi-head attention, with boolean attention masks.
module Synapse.NN.Layers.Attention (attention, causalMask, paddingMask, splitHeads, mergeHeads, MultiHeadAttention (..), multiHeadAttention) where

import Control.Arrow (Arrow, arr)
import Data.Proxy (Proxy (..))
import GHC.Generics (Generic)
import GHC.TypeNats (Div, KnownNat, Nat, natVal, type (*), type (<=))
import Synapse.NN.Layers.Dense (Dense, dense)
import qualified Synapse.Tensor as T
import Synapse.Tensor.Ops (Needs (Fractional), TensorOps (customVJP))
import Synapse.Tensor.Shapes.Axes (CanReduceNonEmpty)

-- | Allow each query to attend to keys at the same or earlier position.
-- 'True' means allowed, including the diagonal. Rectangular masks align the
-- starts of the query and key sequences; they do not align their ends.
-- Broadcast to @'[batch, heads, queries, keys]@ before passing to 'attention'.
causalMask :: (KnownNat queries, KnownNat keys) => T.Tensor '[queries, keys] Bool
causalMask = T.fromGen $ \ix -> T.coordinate @1 ix T.<= T.coordinate @0 ix

-- | Reshape a key-validity mask from @'[batch, keys]@ to @'[batch, 1, 1, keys]@.
-- 'True' marks a real token and 'False' marks padding. Broadcast this result
-- to the attention score shape. Combine it with a broadcast 'causalMask'
-- using @T.&&@ when both restrictions apply.
paddingMask :: (KnownNat batch, KnownNat keys) => T.Tensor '[batch, keys] Bool -> T.Tensor '[batch, 1, 1, keys] Bool
paddingMask = T.reshape

-- | Split batch-first feature vectors into attention heads.
-- Maps @'[batch, time, features]@ to
-- @'[batch, heads, time, Div features heads]@, preserving feature order
-- within each head. The number of heads must be positive and divide the
-- feature width exactly; both conditions are checked at compile time.
--
-- @
-- let query = NN.splitHeads \@2 $ NN.dense queryProjection input
-- @
splitHeads ::
  forall heads {t} batch time features a.
  ( KnownNat batch,
    KnownNat heads,
    KnownNat time,
    KnownNat features,
    KnownNat (Div features heads),
    1 <= heads,
    features ~ (heads * Div features heads),
    T.TensorOps t a
  ) =>
  t '[batch, time, features] a ->
  t '[batch, heads, time, Div features heads] a
splitHeads = T.permuteAxes @'[0, 2, 1, 3] . T.reshape @'[batch, time, heads, Div features heads]

-- | Concatenate attention heads back into batch-first feature vectors.
-- Maps @'[batch, heads, time, depth]@ to @'[batch, time, heads * depth]@.
-- This is the inverse of 'splitHeads'; no type arguments are needed when
-- the input shape is known. Both operations support tensor differentiation.
mergeHeads ::
  forall {t} batch heads time depth a.
  ( KnownNat batch,
    KnownNat heads,
    KnownNat time,
    KnownNat depth,
    KnownNat (heads * depth),
    1 <= heads,
    T.TensorOps t a
  ) =>
  t '[batch, heads, time, depth] a ->
  t '[batch, time, heads * depth] a
mergeHeads = T.reshape @'[batch, time, heads * depth] . T.permuteAxes @'[0, 2, 1, 3]

-- | Normalize finite scores over allowed keys. Empty rows have zero weights
-- and zero cotangents, without introducing an undefined softmax intermediate.
maskedSoftmax ::
  forall batch heads queries keys t a.
  ( KnownNat batch,
    KnownNat heads,
    KnownNat queries,
    KnownNat keys,
    1 <= keys,
    CanReduceNonEmpty '[3] '[batch, heads, queries, keys],
    T.TensorOps t a,
    RealFloat a,
    Floating (T.TensorElement a),
    T.Ord (T.TensorElement a)
  ) =>
  T.Tensor '[batch, heads, queries, keys] Bool ->
  t '[batch, heads, queries, keys] a ->
  t '[batch, heads, queries, keys] a
maskedSoftmax mask = customVJP Fractional normalize (\x -> let weights = normalize x in (weights, weights)) backward
  where
    expand :: T.Tensor '[batch, heads, queries] a -> T.Tensor '[batch, heads, queries, keys] a
    expand = T.broadcast . T.reshape @'[batch, heads, queries, 1]
    normalize scores =
      let valid = T.broadcast $ T.reshape @'[batch, heads, queries, 1] $ T.anyAxes @'[3] mask
          masked = T.select mask scores $ T.full $ negate $ 1 / 0
          safe = T.select valid masked 0
       in T.select mask (T.softmax @3 safe) 0
    backward weights seed = weights * (seed - expand (T.sumAxes @'[3] $ weights * seed))

-- | Compute @softmax (Q K^T / sqrt depth) V@ over the key axis.
-- Inputs have shapes @'[batch, heads, queries, depth]@,
-- @'[batch, heads, keys, depth]@ and @'[batch, heads, keys, values]@.
-- Query and key depths must match and be positive; at least one key is required.
-- Query count and value width may differ from key count and depth.
--
-- 'Nothing' allows every key. A mask has score shape
-- @'[batch, heads, queries, keys]@ and uses 'True' for allowed connections.
-- Fully masked rows produce zero outputs and gradients. Scores and values
-- should be finite. Masked query positions may need a separate output or loss
-- mask: a key padding mask alone does not suppress those query positions.
-- No dropout is applied; training and inference use the same computation.
attention ::
  forall batch heads queries keys depth values t a.
  ( KnownNat batch,
    KnownNat heads,
    KnownNat queries,
    KnownNat keys,
    KnownNat depth,
    KnownNat values,
    1 <= keys,
    CanReduceNonEmpty '[3] '[batch, heads, queries, keys],
    1 <= depth,
    T.TensorOps t a,
    RealFloat a,
    Floating (T.TensorElement a),
    T.Ord (T.TensorElement a),
    Fractional (t '[batch, heads, queries, keys] a)
  ) =>
  Maybe (T.Tensor '[batch, heads, queries, keys] Bool) ->
  t '[batch, heads, queries, depth] a ->
  t '[batch, heads, keys, depth] a ->
  t '[batch, heads, keys, values] a ->
  t '[batch, heads, queries, values] a
attention mask query key value =
  let scale = sqrt $ fromIntegral $ natVal $ Proxy @depth
      scores = T.matmul query (T.transpose key) / T.full scale
      weights = case mask of
        Nothing -> T.softmax @3 scores
        Just allowed -> maskedSoftmax allowed scores
   in T.matmul weights value

-- | Four affine projections shared across all sequence positions.
-- The feature width must be divisible by the number of heads when applied.
-- Initialize projections with 'Dense' parameters and the usual NN initializers.
data MultiHeadAttention (features :: Nat) (heads :: Nat) (t :: T.TensorLike) a = MultiHeadAttention
  { -- | Projection of query feature vectors.
    attentionQuery :: Dense features features t a,
    -- | Projection of key feature vectors.
    attentionKey :: Dense features features t a,
    -- | Projection of value feature vectors.
    attentionValue :: Dense features features t a,
    -- | Projection of concatenated head outputs back to the feature width.
    attentionOutput :: Dense features features t a
  }
  deriving (Generic)

type instance T.WithTensor t (MultiHeadAttention features heads T.Tensor a) = MultiHeadAttention features heads t a

-- | Apply projected attention to batch-first query, key and value sequences.
-- Pass @(input, input, input)@ for self-attention. Cross-attention uses the
-- decoder sequence as queries and the encoder sequence as keys and values.
-- Query and key lengths may differ; all inputs use the parameter feature width.
--
-- Masks follow 'attention': 'True' allows a connection and 'Nothing' allows all.
-- A fully masked row is zero before the output projection, which may add bias.
-- Dropout and positional information are supplied separately.
-- Works as an ordinary function or within any 'Arrow'.
--
-- @
-- let allowed = T.broadcast $ NN.causalMask \@time \@time
-- NN.multiHeadAttention parameters (Just allowed) (input, input, input)
-- @
multiHeadAttention ::
  forall features heads batch queries keys t a {arrow}.
  ( Arrow arrow,
    KnownNat features,
    KnownNat heads,
    KnownNat batch,
    KnownNat queries,
    KnownNat keys,
    KnownNat (Div features heads),
    1 <= heads,
    1 <= Div features heads,
    1 <= keys,
    CanReduceNonEmpty '[3] '[batch, heads, queries, keys],
    features ~ (heads * Div features heads),
    T.TensorOps t a,
    RealFloat a,
    Floating (T.TensorElement a),
    T.Ord (T.TensorElement a),
    Num (t '[batch, queries, features] a),
    Num (t '[batch, keys, features] a),
    Fractional (t '[batch, heads, queries, keys] a)
  ) =>
  MultiHeadAttention features heads t a ->
  Maybe (T.Tensor '[batch, heads, queries, keys] Bool) ->
  arrow
    (t '[batch, queries, features] a, t '[batch, keys, features] a, t '[batch, keys, features] a)
    (t '[batch, queries, features] a)
multiHeadAttention parameters mask = arr $ \(query, key, value) ->
  let q = splitHeads @heads $ dense (attentionQuery parameters) query
      k = splitHeads @heads $ dense (attentionKey parameters) key
      v = splitHeads @heads $ dense (attentionValue parameters) value
      attended = attention mask q k v
   in dense (attentionOutput parameters) $ mergeHeads attended
