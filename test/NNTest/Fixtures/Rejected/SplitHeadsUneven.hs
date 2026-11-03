{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SplitHeadsUneven where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: T.Tensor '[1, 2, 3] Double -> T.Tensor '[1, 2, 2, 1] Double
invalid = NN.splitHeads @2
