{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ComparisonShape where

import qualified Synapse.Tensor as T

bad = T.zeros @'[2, 3] @Int T.== T.zeros @'[3, 2]
