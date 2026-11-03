{-# LANGUAGE DataKinds #-}

module MergeHeadsShape where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: T.Tensor '[1, 2, 3, 4] Double -> T.Tensor '[1, 3, 7] Double
invalid = NN.mergeHeads
