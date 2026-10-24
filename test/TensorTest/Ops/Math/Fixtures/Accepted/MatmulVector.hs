{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MatmulVector where

import qualified Synapse.Tensor as T

vectorProduct :: T.Tensor '[4] Int
vectorProduct = T.matmul (T.ones @'[3]) (T.ones @'[3, 4])
