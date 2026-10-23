-- | Neural network functions.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.NN as NN
-- @
module Synapse.NN (mse, mae, huber, smoothL1) where

import Synapse.NN.Loss (huber, mae, mse, smoothL1)
