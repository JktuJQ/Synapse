{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module DenseOutput where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: T.Tensor '[2, 4] Double
bad = NN.forward (NN.Dense (T.zeros @'[2, 3]) (T.zeros @'[3])) (T.zeros @'[2, 2])
