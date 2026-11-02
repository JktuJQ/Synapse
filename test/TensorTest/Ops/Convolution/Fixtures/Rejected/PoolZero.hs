{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PoolZero where

import qualified Synapse.Tensor as T

bad = T.avgPool2d @'[2, 2] @'[0, 1] @'[0, 0, 0, 0] True (T.ones @'[1, 1, 2, 2] @Double)
