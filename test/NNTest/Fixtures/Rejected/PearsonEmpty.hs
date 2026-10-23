{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PearsonEmpty where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: T.Tensor '[] Double
bad = NN.pearson (T.zeros @'[0]) (T.zeros @'[0])
