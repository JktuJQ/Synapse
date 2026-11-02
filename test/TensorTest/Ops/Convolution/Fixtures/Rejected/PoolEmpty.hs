{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PoolEmpty where

import qualified Synapse.Tensor as T

bad = T.maxPool2d @'[2, 2] @'[1, 1] @'[1, 1, 1, 1] @'[1, 1] (T.zeros @'[1, 1, 0, 2] @Double)
