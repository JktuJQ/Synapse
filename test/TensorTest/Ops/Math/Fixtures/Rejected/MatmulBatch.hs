{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulBatch where

import qualified Synapse.Tensor as T

bad = T.matmul (T.ones @'[2, 3, 4] @Int) (T.ones @'[3, 4, 5])
