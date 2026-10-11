{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulVector where

import qualified Synapse.Tensor as T

bad = T.matmul (T.ones @'[3] @Int) (T.ones @'[3, 4])
