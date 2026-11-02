{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ZeroKernel where

import qualified Synapse.Tensor as T

bad = T.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] @1 (T.ones @'[1, 1, 2, 2] @Double) (T.ones @'[1, 1, 0, 1])
