{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PatchDilation where

import qualified Synapse.Tensor as T

bad = T.extractPatches2d @'[1, 1] @'[1, 1] @'[0, 0, 0, 0] @'[0, 1] (T.ones @'[1, 1, 2, 2] @Double)
