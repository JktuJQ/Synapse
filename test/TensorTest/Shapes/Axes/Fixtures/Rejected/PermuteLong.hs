{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PermuteLong where

import qualified Synapse.Tensor as T

bad = T.permuteAxes @'[0, 1, 2, 3] (T.zeros @'[2, 3, 4] @Int)
