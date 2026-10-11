{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ConcatAxis where

import qualified Synapse.Tensor as T

bad = T.concat @2 (T.zeros @'[2, 3] @Int) (T.zeros @'[2, 3])
