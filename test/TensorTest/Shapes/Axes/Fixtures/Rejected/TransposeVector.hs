{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TransposeVector where

import qualified Synapse.Tensor as T

bad = T.transpose (T.zeros @'[3] @Int)
