{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module DotMatrix where

import qualified Synapse.Tensor as T

bad = T.dot (T.ones @'[2, 3] @Int) (T.ones @'[3])
