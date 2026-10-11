{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ConcatPrefix where

import qualified Synapse.Tensor as T

bad = T.concat @1 (T.zeros @'[2, 3] @Int) (T.zeros @'[4, 3])
