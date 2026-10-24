{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulVectorInner where

import qualified Synapse.Tensor as T

bad = T.matmul (T.ones @'[3] @Int) (T.ones @'[4])
