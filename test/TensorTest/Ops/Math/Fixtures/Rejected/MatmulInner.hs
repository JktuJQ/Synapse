{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulInner where

import qualified Synapse.Tensor as T

bad = T.matmul (T.ones @'[2, 3] @Int) (T.ones @'[4, 5])
