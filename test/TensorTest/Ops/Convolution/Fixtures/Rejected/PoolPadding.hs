{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PoolPadding where

import qualified Synapse.Tensor as T

bad = T.maxPool2d @'[2, 2] @'[1, 1] @'[2, 0, 0, 0] @'[1, 1] (T.ones @'[1, 1, 2, 2] @Double)
