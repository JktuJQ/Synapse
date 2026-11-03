{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Learnable vectors selected by integer vocabulary indices.
module Synapse.NN.Layers.Embedding (Embedding (..), embedding) where

import Control.Arrow (Arrow, arr)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat)
import Synapse.Tensor.Ops (TensorOps)
import Synapse.Tensor.Ops.Index (unsafeLookupRows)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Index (LookupRowsShape)
import Synapse.Tensor.Type (Tensor, TensorLike, WithTensor)

-- | One learnable vector per vocabulary entry.
-- Weights have shape @'[vocabulary, features]@ and can be initialized with the ordinary NN initializers.
newtype Embedding (vocabulary :: Nat) (features :: Nat) (t :: TensorLike) a = Embedding
  { -- | The vocabulary table, with one vector per row.
    embeddingWeights :: t '[vocabulary, features] a
  }
  deriving (Generic)

type instance WithTensor t (Embedding vocabulary features Tensor a) = Embedding vocabulary features t a

-- | Replace each integer index with its vocabulary vector.
-- For example, indices of shape @'[batch, time]@ produce @'[batch, time, features]@.
-- Index values must lie in @[0, vocabulary)@; they are unchecked and invalid values may cause invalid memory access.
-- Gradients are dense: repeated indices add their contributions, and unselected rows receive zeros.
-- The ordinary index tensor is not differentiated.
-- Supports ordinary function application and composition within an arbitrary arrow.
embedding ::
  ( Arrow arrow,
    TensorOps t a,
    Shape '[vocabulary, features],
    Shape indices,
    Shape (LookupRowsShape indices features)
  ) =>
  Embedding vocabulary features t a ->
  arrow (Tensor indices Int) (t (LookupRowsShape indices features) a)
embedding parameters = arr $ unsafeLookupRows $ embeddingWeights parameters
