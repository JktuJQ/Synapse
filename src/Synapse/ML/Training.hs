{- | This module provides interface that is related to neural networks training.
-}


module Synapse.ML.Training
    ( -- * Re-exports
      module Synapse.ML.Training.Losses
    , module Synapse.ML.Training.Metrics

    , module Synapse.ML.Training.LearningRates
    , module Synapse.ML.Training.Batching
    ) where

import Synapse.ML.Training.Losses
import Synapse.ML.Training.Metrics

import Synapse.ML.Training.LearningRates
import Synapse.ML.Training.Batching
