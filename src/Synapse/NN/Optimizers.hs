-- | Pure optimizers with explicit state and tensorwise parameter updates.
-- Hyperparameters are supplied by the caller without runtime validation.
-- Optimizer steps use the strict 'State' type from
-- <https://hackage.haskell.org/package/transformers/docs/Control-Monad-Trans-State-Strict.html transformers>.
-- Run actions with @runState@ or @evalState@, available from "Synapse.NN.Forward".
module Synapse.NN.Optimizers
  ( module Synapse.NN.Optimizers.Type,
    module Synapse.NN.Optimizers.SGD,
    module Synapse.NN.Optimizers.Adam,
    module Synapse.NN.Optimizers.RMSprop,
  )
where

import Synapse.NN.Optimizers.Adam
import Synapse.NN.Optimizers.RMSprop
import Synapse.NN.Optimizers.SGD
import Synapse.NN.Optimizers.Type
