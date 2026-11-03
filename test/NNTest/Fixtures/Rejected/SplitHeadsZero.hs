{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SplitHeadsZero where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: T.Tensor '[1, 2, 4] Double -> T.Tensor '[1, 0, 2, 0] Double
invalid = NN.splitHeads @0
