{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Positional information for sequence models.
module Synapse.NN.Layers.Position (sinusoidalPositions, rope) where

import Control.Arrow (Arrow, arr)
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat, Mod, natVal)
import qualified Synapse.Tensor as T
import Synapse.Tensor.Ops (TensorOps (literal))

-- | Generate sinusoidal positions of shape @'[time, features]@, starting at
-- the supplied position offset. For feature pair @2i, 2i+1@ the entries are
-- @sin (position / 10000^(2i/features))@ and @cos (position / 10000^(2i/features))@.
-- Odd feature widths retain the last sine entry. Empty dimensions are allowed.
-- Broadcast to @'[batch, time, features]@ and add to token embeddings.
-- Positions are constants with respect to differentiation.
--
-- For learned positions, use an 'Synapse.NN.Layers.Embedding.Embedding' table
-- and apply @embedding@ to integer position indices instead.
--
-- @
-- input + T.broadcast (NN.sinusoidalPositions \@time \@features 0)
-- @
sinusoidalPositions ::
  forall time features t a.
  ( KnownNat time,
    KnownNat features,
    T.TensorOps t a,
    Floating a,
    Floating (T.TensorElement a),
    T.FromIntegral (T.TensorElement Int) (T.TensorElement a)
  ) =>
  Int ->
  t '[time, features] a
sinusoidalPositions offset = T.fromGen $ \ix ->
  let feature = T.coordinate @1 ix
      position :: T.TensorElement a
      position = T.fromIntegral (T.coordinate @0 ix) + T.element (fromIntegral offset)
      paired :: T.TensorElement a
      paired = T.fromIntegral $ 2 * T.quot feature 2
      width :: T.TensorElement a
      width = T.element $ fromIntegral $ max 1 $ natVal $ Proxy @features
      angle :: T.TensorElement a
      angle = position / (10000 ** (paired / width))
   in T.select (T.rem feature 2 T.== 0) (sin angle) (cos angle)

-- | Apply <https://arxiv.org/abs/2104.09864 rotary positional embedding>
-- to projected queries or keys of shape @'[batch, heads, time, features]@.
-- Adjacent features @(0,1), (2,3), ...@ form rotation pairs; this is the
-- interleaved convention, not the split-half convention used by some models.
-- The feature width must be even, checked at compile time.
--
-- The arguments are the positive frequency base (commonly @10000@) and the
-- starting position offset. For pair @2i, 2i+1@ at position @p@, with
-- @angle = p / base^(2i/features)@, the output is:
--
-- @
-- (x * cos angle - y * sin angle, x * sin angle + y * cos angle)
-- @
--
-- Apply to queries and keys after their projections and head splitting,
-- before 'Synapse.NN.Layers.Attention.attention'; values are left unchanged.
-- Query and key offsets may differ when processing cached or chunked inputs.
-- Rotations preserve vector norms and gradients flow to the input tensor.
-- Empty batch, head, time and feature dimensions are supported.
--
-- @
-- NN.attention mask (NN.rope 10000 offset query) (NN.rope 10000 offset key) value
-- @
rope ::
  forall batch heads time features t a {arrow}.
  ( Arrow arrow,
    KnownNat batch,
    KnownNat heads,
    KnownNat time,
    KnownNat features,
    Mod features 2 ~ 0,
    T.TensorOps t a,
    Floating a,
    Floating (T.TensorElement a),
    T.FromIntegral (T.TensorElement Int) (T.TensorElement a),
    Num (t '[batch, heads, time, features] a)
  ) =>
  a ->
  Int ->
  arrow (t '[batch, heads, time, features] a) (t '[batch, heads, time, features] a)
rope base offset = arr $ \input ->
  let angles :: T.Tensor '[time, features] a
      angles = T.fromGen $ \ix ->
        let position :: T.TensorElement a
            position = T.fromIntegral (T.coordinate @0 ix) + T.element (fromIntegral offset)
            paired :: T.TensorElement a
            paired = T.fromIntegral $ 2 * T.quot (T.coordinate @1 ix) 2
            width = T.element $ fromIntegral $ max 1 $ natVal $ Proxy @features
         in position / (T.element base ** (paired / width))
      cosines = literal $ T.broadcast $ cos angles
      sines = literal $ T.broadcast $ sin angles
      partners = T.fromGen @'[batch, heads, time, features] $ \ix ->
        let feature = T.coordinate @3 ix
         in T.select (T.rem feature 2 T.== 0) (feature + 1) (feature - 1)
      signs = literal $ T.broadcast $ T.fromGen @'[features] $ \ix ->
        T.select (T.rem (T.coordinate @0 ix) 2 T.== 0) (-1) 1
   in input * cosines + T.unsafeGather @3 input partners * signs * sines
