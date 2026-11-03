{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmbeddingIndices where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

indices :: T.Tensor '[2] Double
indices = T.fromList [0, 1]

bad :: T.Tensor '[2, 2] Double
bad = NN.embedding (NN.Embedding $ T.ones @'[4, 2]) indices
