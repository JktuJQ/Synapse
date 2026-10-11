{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module DotLength where

import qualified Synapse.Tensor as T

bad = T.dot (T.ones @'[3] @Int) (T.ones @'[4])
