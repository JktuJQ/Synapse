{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module GroupsZero where

import qualified Synapse.Tensor as T

bad = T.conv1d @'[1] @'[0, 0] @'[1] @0 (T.ones @'[1, 2, 3] @Double) (T.ones @'[2, 2, 2])
