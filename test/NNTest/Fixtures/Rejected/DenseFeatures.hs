{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module DenseFeatures where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: T.Tensor '[0, 3] Double
bad = NN.dense (NN.Dense (T.zeros @'[2, 3]) (T.zeros @'[3])) (T.zeros @'[0, 5])
