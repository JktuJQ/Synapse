{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ArgminAxis where

import qualified Synapse.Tensor as T

bad = T.argmin @2 (T.zeros @'[2, 3] @Int)
