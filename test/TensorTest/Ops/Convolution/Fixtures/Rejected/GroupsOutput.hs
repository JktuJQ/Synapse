{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module GroupsOutput where

import qualified Synapse.Tensor as T

bad = T.conv1d @'[1] @'[0, 0] @'[1] @2 (T.ones @'[1, 4, 3] @Double) (T.ones @'[3, 2, 2])
