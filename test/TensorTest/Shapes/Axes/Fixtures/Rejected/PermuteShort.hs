{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module PermuteShort where

import qualified Synapse.Tensor as T

bad = T.permuteAxes @'[0, 1] (T.zeros @'[2, 3, 4] @Int)
