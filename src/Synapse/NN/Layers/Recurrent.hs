-- | Single-layer recurrent parameters, cells and fixed-length sequence functions.
-- Sequence inputs and outputs use time-major layout. Parameters are ordinary
-- tensor records, separate from the explicitly supplied recurrent state.
-- Each function applies one layer in one direction; compose layer functions
-- to build a deeper model. Cells consume one batched timestep, while sequence
-- functions return all timesteps and update the final state.
--
-- @
-- import qualified Synapse.NN as NN
-- import qualified Synapse.Tensor as T
--
-- -- Three input features, eight hidden features, two examples per batch.
-- parameters :: NN.LSTM 3 8 T.Tensor Float
-- parameters = NN.LSTM
--   (T.full \@'[3, 32] 0.1) (T.full \@'[8, 32] 0.1)
--   (T.zeros \@'[32]) (T.zeros \@'[32])
-- initial = NN.LSTMState (T.zeros \@'[2, 8]) (T.zeros \@'[2, 8])
-- inputs = T.ones \@'[5, 2, 3]
-- (outputs, finalState) = NN.runState (NN.lstm parameters inputs) initial
-- @
--
-- Initialize the weight fields with "Synapse.NN.Initialization" for training.
-- Include the initial state among differentiation inputs when its gradient is
-- needed; it is never detached automatically between steps or sequence calls.
module Synapse.NN.Layers.Recurrent
  ( module Synapse.NN.Layers.Recurrent.RNN,
    module Synapse.NN.Layers.Recurrent.GRU,
    module Synapse.NN.Layers.Recurrent.LSTM,
  )
where

import Synapse.NN.Layers.Recurrent.GRU
import Synapse.NN.Layers.Recurrent.LSTM
import Synapse.NN.Layers.Recurrent.RNN
