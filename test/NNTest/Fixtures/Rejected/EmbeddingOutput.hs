{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmbeddingOutput where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: T.Tensor '[2, 2, 3] Double
bad = NN.embedding (NN.Embedding $ T.ones @'[4, 2]) $ T.fromList @'[2, 2] [0, 1, 2, 3]
