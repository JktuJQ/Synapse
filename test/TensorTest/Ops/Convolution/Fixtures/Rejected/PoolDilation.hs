{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PoolDilation where

import qualified Synapse.Tensor as T

bad = T.maxPool2d @'[1, 1] @'[1, 1] @'[0, 0, 0, 0] @'[0, 1] (T.ones @'[1, 1, 2, 2] @Double)
