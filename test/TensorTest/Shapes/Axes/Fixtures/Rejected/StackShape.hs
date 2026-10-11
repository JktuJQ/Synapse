{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module StackShape where

import qualified Synapse.Tensor as T

bad = T.stack @0 (T.zeros @'[2, 3] @Int) (T.zeros @'[2, 4])
