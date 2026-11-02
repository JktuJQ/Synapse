{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module DilatedKernel where

import qualified Synapse.Tensor as T

bad = T.conv1d @'[1] @'[0, 0] @'[3] @1 (T.ones @'[1, 1, 3] @Double) (T.ones @'[1, 1, 2])
