{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TransposeGroups where

import qualified Synapse.Tensor as T

bad = T.convTranspose1d @'[1] @'[0, 0] @'[1] @2 @'[0] (T.ones @'[1, 3, 3] @Double) (T.ones @'[3, 1, 2])
