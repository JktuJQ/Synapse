{-# LANGUAGE DataKinds #-}

module RoPEWidth where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: T.Tensor '[1, 2, 3, 5] Double -> T.Tensor '[1, 2, 3, 5] Double
invalid = NN.rope 10000 0
