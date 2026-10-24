{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulRightScalar where

import qualified Synapse.Tensor as T

bad = T.matmul (T.ones @'[3, 4]) (T.scalar (T.constant (1 :: Int)))
