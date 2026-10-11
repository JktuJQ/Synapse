{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SelectShape where

import qualified Synapse.Tensor as T

bad = T.select (T.full @'[2] True) (T.zeros @'[2] @Int) (T.ones @'[3])
