{-# LANGUAGE DataKinds #-}

module GRUState where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: NN.GRU 2 3 T.Tensor Double -> T.Tensor '[4, 1, 2] Double -> NN.State (T.Tensor '[1, 4] Double) (T.Tensor '[4, 1, 3] Double)
bad = NN.gru
