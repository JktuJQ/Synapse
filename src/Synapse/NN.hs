-- | Neural network functions.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.NN as NN
-- @
module Synapse.NN
  ( ExecutionMode (..),
    extractWeights,
    loadWeights,
    Dataset (..),
    getSample,
    loadBatches,
    LastBatch (..),
    Batch (..),
    SomeBatch (..),
    shuffle,
    withBatch,
    batchSize,
    batchMask,
    module Synapse.NN.Layers,
    module Synapse.NN.Optimizers,
    mse,
    mae,
    huber,
    smoothL1,
    binaryCrossEntropy,
    binaryCrossEntropyWithLogits,
    negativeLogLikelihood,
    crossEntropy,
    unsafeNegativeLogLikelihood,
    unsafeCrossEntropy,
    smoothLabels,
    unsafeCrossEntropySmoothed,
    rmse,
    r2,
    binaryAccuracy,
    accuracy,
    Averaging (..),
    precision,
    recall,
    f1,
    precisionPerClass,
    recallPerClass,
    f1PerClass,
    confusionMatrix,
    binaryPrecision,
    binaryRecall,
    binaryF1,
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
import Synapse.NN.Execution (ExecutionMode (..))
import Synapse.NN.Initialization (kaimingNormal, kaimingUniform, normal, uniform, xavierNormal, xavierUniform)
import Synapse.NN.Layers
import Synapse.NN.Loss (binaryCrossEntropy, binaryCrossEntropyWithLogits, crossEntropy, huber, mae, mse, negativeLogLikelihood, smoothL1, smoothLabels, unsafeCrossEntropy, unsafeCrossEntropySmoothed, unsafeNegativeLogLikelihood)
import Synapse.NN.Metrics (Averaging (..), accuracy, binaryAccuracy, binaryF1, binaryPrecision, binaryRecall, confusionMatrix, f1, f1PerClass, precision, precisionPerClass, r2, recall, recallPerClass, rmse)
import Synapse.NN.Optimizers
import Synapse.NN.Weights (extractWeights, loadWeights)
import Prelude hiding (sin, tanh)
