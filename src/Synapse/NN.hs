-- | Neural network functions.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.NN as NN
-- @
module Synapse.NN (mse, mae, huber, smoothL1, relu, leakyRelu, sigmoid, tanh, softplus) where

import Synapse.NN.Activation (leakyRelu, relu, sigmoid, softplus, tanh)
import Synapse.NN.Loss (huber, mae, mse, smoothL1)
import Prelude hiding (tanh)
