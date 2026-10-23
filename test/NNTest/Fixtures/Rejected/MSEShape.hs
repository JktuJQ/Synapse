{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MSEShape where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: T.Tensor '[] Double
bad = NN.mse T.mean (T.ones @'[2] @Double) (T.ones @'[3] @Double)
