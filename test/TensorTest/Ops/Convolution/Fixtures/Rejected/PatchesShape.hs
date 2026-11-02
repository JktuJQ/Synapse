{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PatchesShape where

import qualified Synapse.Tensor as T

bad = T.combinePatches2d @'[1, 1, 2, 2] @'[2, 2] @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] (T.ones @'[1, 3, 1] @Double)
