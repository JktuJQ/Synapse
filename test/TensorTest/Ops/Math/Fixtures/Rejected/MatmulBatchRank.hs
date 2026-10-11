{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulBatchRank where

import qualified Synapse.Tensor as T

bad = T.matmul (T.ones @'[1, 3, 4] @Int) (T.ones @'[4, 5])
