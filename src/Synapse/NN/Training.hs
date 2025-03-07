{- | This module provides interface that is related to neural networks training.
-}


module Synapse.NN.Training
    ( -- * Re-exports
      module Synapse.NN.Training.Losses
    , module Synapse.NN.Training.Metrics

    , module Synapse.NN.Training.LearningRates
    , module Synapse.NN.Training.Batching

    , module Synapse.NN.Training.Optimizers
    ) where

import Synapse.NN.Training.Losses
import Synapse.NN.Training.Metrics

import Synapse.NN.Training.LearningRates
import Synapse.NN.Training.Batching

import Synapse.NN.Training.Optimizers
