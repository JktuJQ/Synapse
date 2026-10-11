{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PermuteDuplicate where

import qualified Synapse.Tensor as T

bad = T.permuteAxes @'[1, 1, 0] (T.zeros @'[2, 3, 4] @Int)
