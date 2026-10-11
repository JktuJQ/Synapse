{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module StackAxis where

import qualified Synapse.Tensor as T

bad = T.stack @3 (T.zeros @'[2, 3] @Int) (T.zeros @'[2, 3])
