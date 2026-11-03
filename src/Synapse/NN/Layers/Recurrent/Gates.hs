{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | Packing and splitting recurrent gate projections.
module Synapse.NN.Layers.Recurrent.Gates (split3, split4) where

import GHC.TypeNats (KnownNat, type (*))
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Shape as Pullback
import Synapse.Tensor.Ops (Needs (Numeric), TensorOps (customVJP))
import qualified Synapse.Tensor.Ops.Primitives.Shape as Raw

-- | Split reset, update and candidate gate projections without retaining a cache.
split3 :: forall hidden batch t a. (KnownNat batch, KnownNat hidden, KnownNat (3 * hidden), TensorOps t a) => t '[batch, 3 * hidden] a -> (t '[batch, hidden] a, t '[batch, hidden] a, t '[batch, hidden] a)
split3 = customVJP Numeric forward (\input -> (forward input, ())) (\() (r, z, n) -> Pullback.joinMatrixChunks @3 @hidden [r, z, n])
  where
    forward input = (Raw.matrixChunk @hidden 0 input, Raw.matrixChunk @hidden 1 input, Raw.matrixChunk @hidden 2 input)

-- | Split input, forget, candidate and output gate projections.
split4 :: forall hidden batch t a. (KnownNat batch, KnownNat hidden, KnownNat (4 * hidden), TensorOps t a) => t '[batch, 4 * hidden] a -> (t '[batch, hidden] a, t '[batch, hidden] a, t '[batch, hidden] a, t '[batch, hidden] a)
split4 = customVJP Numeric forward (\input -> (forward input, ())) (\() (i, f, g, o) -> Pullback.joinMatrixChunks @4 @hidden [i, f, g, o])
  where
    forward input = (Raw.matrixChunk @hidden 0 input, Raw.matrixChunk @hidden 1 input, Raw.matrixChunk @hidden 2 input, Raw.matrixChunk @hidden 3 input)
