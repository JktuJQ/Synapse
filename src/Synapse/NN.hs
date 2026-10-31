-- | Neural network functions.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.NN as NN
-- @
module Synapse.NN
  ( Dataset (..),
    getSample,
    loadBatches,
    LastBatch (..),
    Batch (..),
    SomeBatch (..),
    shuffle,
    withBatch,
    batchSize,
    batchMask,
    Dense (..),
    dense,
    Optimizer (..),
    SGD (..),
    updateParameters,
    mse,
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
    sin,
    softplus,
    uniform,
    normal,
    xavierUniform,
    xavierNormal,
    kaimingUniform,
    kaimingNormal,
  )
where

import Synapse.NN.Activation (leakyRelu, relu, sigmoid, sin, softplus, tanh)
import Synapse.NN.Data (Batch (..), Dataset (..), LastBatch (..), SomeBatch (..), batchMask, batchSize, getSample, loadBatches, shuffle, withBatch)
import Synapse.NN.Dense (Dense (..), dense)
import Synapse.NN.Initialization (kaimingNormal, kaimingUniform, normal, uniform, xavierNormal, xavierUniform)
import Synapse.NN.Loss (huber, mae, mse, smoothL1)
import Synapse.NN.Metrics (explainedVariance, mape, maxError, meanError, msle, pearson, r2, rae, rmse, rmsle, rse, smape, wmape)
import Synapse.NN.Optimizer (Optimizer (..), SGD (..), updateParameters)
import Prelude hiding (sin, tanh)
