{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MaxErrorEmpty where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: T.Tensor '[] Double
bad = NN.maxError (T.zeros @'[0]) (T.zeros @'[0])
