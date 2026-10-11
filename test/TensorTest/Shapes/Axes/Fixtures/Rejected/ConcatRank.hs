{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ConcatRank where

import qualified Synapse.Tensor as T

bad = T.concat @0 (T.zeros @'[2, 3] @Int) (T.zeros @'[2])
