{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulScalar where

import qualified Synapse.Tensor as T

bad = T.matmul (T.scalar (T.element (1 :: Int))) (T.ones @'[3, 4])
