-- | Execution settings for neural network functions.
module Synapse.NN.Execution (ExecutionMode (..)) where

-- | Select training or evaluation behavior for operations such as BatchNorm and dropout.
data ExecutionMode = Train | Eval deriving (Eq, Show)
