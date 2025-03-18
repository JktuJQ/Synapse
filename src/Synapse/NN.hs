{- | This module provides functions and datatypes that are needed in work with neural networks.

"Synapse" tries to supply as much of the common interface of any neural network library,
such as the collection of different models, layers, optimisers, training algorithms and etc.
-}


module Synapse.NN 
    ( -- * Re-exports

      module Synapse.NN.Layers

    , module Synapse.NN.Models
    , module Synapse.NN.Training
    , module Synapse.NN.Batching

    , module Synapse.NN.Losses
    , module Synapse.NN.Metrics

    , module Synapse.NN.LearningRates
    
    , module Synapse.NN.Optimizers
    ) where


import Synapse.NN.Layers

import Synapse.NN.Models
import Synapse.NN.Training
import Synapse.NN.Batching

import Synapse.NN.Losses
import Synapse.NN.Metrics

import Synapse.NN.LearningRates

import Synapse.NN.Optimizers
