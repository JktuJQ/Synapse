{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Conv3Channels where

import qualified Synapse.Tensor as T

bad = T.conv3d @'[1, 1, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1] @1 (T.ones @'[1, 2, 2, 2, 2] @Double) (T.ones @'[1, 3, 1, 1, 1])
