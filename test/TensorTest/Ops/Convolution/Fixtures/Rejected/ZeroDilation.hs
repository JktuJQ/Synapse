{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ZeroDilation where

import qualified Synapse.Tensor as T

bad = T.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[0, 1] @1 (T.ones @'[1, 1, 3, 3] @Double) (T.ones @'[1, 1, 2, 2])
