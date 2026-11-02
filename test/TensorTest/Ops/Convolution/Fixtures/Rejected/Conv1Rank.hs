{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Conv1Rank where

import qualified Synapse.Tensor as T

bad = T.conv1d @'[1] @'[0, 0] @'[1] @1 (T.ones @'[1, 1, 2, 2] @Double) (T.ones @'[1, 1, 2])
