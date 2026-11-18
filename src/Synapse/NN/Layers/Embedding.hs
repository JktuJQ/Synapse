{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoDeepSubsumption #-}

-- | Learnable vectors selected by integer vocabulary indices.
module Synapse.NN.Layers.Embedding (Embedding (..), embedding) where

import Control.Arrow (Arrow, arr)
import GHC.Generics (Generic)
import GHC.TypeLits (Nat)
import Synapse.Tensor.Ops.Definition (TensorOps)
import Synapse.Tensor.Ops.Index (unsafeLookupRows)
import Synapse.Tensor.Representation (Tensor, TensorLike)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Index (LookupRowsShape)
import Synapse.Tensor.Structure (TensorStructure, WithTensor)

{- | One learnable vector per vocabulary entry.

Weights have shape @'[vocabulary, features]@ and can be initialized with the ordinary NN initializers.
-}
newtype Embedding (vocabulary :: Nat) (features :: Nat) (t :: TensorLike) a = Embedding
    { embeddingWeights :: t '[vocabulary, features] a
    }
    deriving (Generic)

instance TensorStructure (Embedding vocabulary features Tensor a) where
    type WithTensor t (Embedding vocabulary features Tensor a) = Embedding vocabulary features t a

{- | Replace each integer index with its vocabulary vector.

For example, indices of shape @'[batch, time]@ produce @'[batch, time, features]@.
Index values must lie in @[0, vocabulary)@; they are unchecked and invalid values may cause invalid memory access.
Gradients are dense: repeated indices add their contributions, and unselected rows receive zeros.
The ordinary index tensor is not differentiated.
-}
embedding
    :: ( Arrow arrow
       , TensorOps t a
       , Shape '[vocabulary, features]
       , Shape indices
       , Shape (LookupRowsShape indices features)
       )
    => Embedding vocabulary features t a
    -> arrow (Tensor indices Int) (t (LookupRowsShape indices features) a)
embedding parameters = arr $ unsafeLookupRows $ embeddingWeights parameters
