{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmptyLogSoftmax where

import qualified Synapse.Tensor as T

bad = T.logSoftmax @0 (T.zeros @'[0, 3] @Double)
