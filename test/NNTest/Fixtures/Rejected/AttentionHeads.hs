{-# LANGUAGE DataKinds #-}

module AttentionHeads where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: NN.MultiHeadAttention 3 2 T.Tensor Double -> T.Tensor '[1, 2, 3] Double -> T.Tensor '[1, 2, 3] Double
invalid p x = NN.multiHeadAttention p Nothing (x, x, x)
