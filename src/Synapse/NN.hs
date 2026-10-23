-- | Neural network functions.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.NN as NN
-- @
module Synapse.NN
  ( mse,
    mae,
    huber,
    smoothL1,
    rmse,
    r2,
    meanError,
    maxError,
    mape,
    smape,
    wmape,
    msle,
    rmsle,
    rae,
    rse,
    explainedVariance,
    pearson,
    relu,
    leakyRelu,
    sigmoid,
    tanh,
    softplus,
    uniform,
    normal,
    xavierUniform,
    xavierNormal,
    kaimingUniform,
    kaimingNormal,
  )
where

import Synapse.NN.Activation (leakyRelu, relu, sigmoid, softplus, tanh)
import Synapse.NN.Initialization (kaimingNormal, kaimingUniform, normal, uniform, xavierNormal, xavierUniform)
import Synapse.NN.Loss (huber, mae, mse, smoothL1)
import Synapse.NN.Metrics (explainedVariance, mape, maxError, meanError, msle, pearson, r2, rae, rmse, rmsle, rse, smape, wmape)
import Prelude hiding (tanh)
