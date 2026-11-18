{- | Update model parameters using gradients and persistent optimizer state.
An optimizer value stores hyperparameters; 'OptimizerState' stores accumulated statistics.
Initialize the state once with 'initialState' and carry it between calls to 'optimizerStep'.

@
let optimizer = SGD 0.01
    initial = initialState optimizer parameters
    (updated, nextState) = runState (optimizerStep optimizer parameters gradients) initial
@
-}
module Synapse.NN.Optimizers
    ( module Synapse.NN.Optimizers.Optimizer
    , module Synapse.NN.Optimizers.SGD
    , module Synapse.NN.Optimizers.RMSprop
    , module Synapse.NN.Optimizers.Adam
    )
where

import Synapse.NN.Optimizers.Adam
import Synapse.NN.Optimizers.Optimizer
import Synapse.NN.Optimizers.RMSprop
import Synapse.NN.Optimizers.SGD
