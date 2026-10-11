{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SelectMask where

import qualified Synapse.Tensor as T

bad = T.select (T.full @'[3] True) (T.zeros @'[2] @Int) (T.ones @'[2])
