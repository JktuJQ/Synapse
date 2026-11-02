{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Patches3Stride where

import qualified Synapse.Tensor as T

bad = T.extractPatches3d @'[1, 1, 1] @'[1, 0, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1] (T.ones @'[1, 1, 2, 2, 2] @Double)
