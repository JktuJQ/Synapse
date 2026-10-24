{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulEmptyBatch where

import qualified Synapse.Tensor as T

bad = T.matmul (T.ones @'[0, 2, 3] @Int) (T.ones @'[2, 3, 4])
