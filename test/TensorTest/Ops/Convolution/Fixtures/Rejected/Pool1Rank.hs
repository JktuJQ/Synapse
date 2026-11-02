{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Pool1Rank where

import qualified Synapse.Tensor as T

bad = T.maxPool1d @'[2] @'[1] @'[0, 0] @'[1] (T.ones @'[1, 1, 2, 2] @Double)
