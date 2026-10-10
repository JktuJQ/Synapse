{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module DuplicateAxes where

import qualified Synapse.Tensor as T

bad = T.sumAxes @'[1, 1] (T.zeros @'[2, 3] @Int)
