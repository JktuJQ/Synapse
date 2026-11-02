{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Patches1Shape where

import qualified Synapse.Tensor as T

bad = T.combinePatches1d @'[1, 1, 3] @'[2] @'[1] @'[0, 0] @'[1] (T.ones @'[1, 3, 2] @Double)
