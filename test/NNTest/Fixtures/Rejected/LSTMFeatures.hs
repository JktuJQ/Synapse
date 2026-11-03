{-# LANGUAGE DataKinds #-}

module LSTMFeatures where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: NN.LSTM 2 3 T.Tensor Double -> T.Tensor '[4, 1, 5] Double -> NN.State (NN.LSTMState 1 3 T.Tensor Double) (T.Tensor '[4, 1, 3] Double)
bad = NN.lstm
