{-# LANGUAGE TypeFamilies #-}

{- | Pure parameter updates with explicit optimizer state.
Updates build tensor expressions without executing them.
Parameters and gradients must have corresponding fields and tensor shapes.
-}
module Synapse.NN.Optimizers.Optimizer (Optimizer (..)) where

import Control.Monad.Trans.State.Strict (State)

-- | Optimizers for a parameter structure.
class Optimizer optimizer parameters where
    -- | Accumulated values carried between updates; use @()@ when no history is needed.
    type OptimizerState optimizer parameters

    -- | Initialize optimizer state using the structure and shapes of the supplied parameters.
    initialState
        :: optimizer
        -> parameters
        -> OptimizerState optimizer parameters

    -- | Update parameters using their gradients in a pure state computation.
    optimizerStep
        :: optimizer
        -> parameters
        -> parameters
        -> State (OptimizerState optimizer parameters) parameters
