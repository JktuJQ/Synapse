{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module GroupsInput where

import qualified Synapse.Tensor as T

bad = T.conv1d @'[1] @'[0, 0] @'[1] @2 (T.ones @'[1, 3, 3] @Double) (T.ones @'[4, 1, 2])
