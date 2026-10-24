{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulBatchRank where

import qualified Synapse.Tensor as T

bad = T.matmul (T.ones @'[2, 3, 4] @Int) (T.ones @'[3, 5, 4, 6])
