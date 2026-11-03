{-# LANGUAGE DataKinds #-}

module AttentionDepth where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: T.Tensor '[1, 1, 2, 2] Double -> T.Tensor '[1, 1, 3, 4] Double -> T.Tensor '[1, 1, 3, 2] Double -> T.Tensor '[1, 1, 2, 2] Double
invalid = NN.attention Nothing
