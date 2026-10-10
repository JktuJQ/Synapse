{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmptyMinimum where

import qualified Synapse.Tensor as T

bad = T.minimum (T.zeros @'[2, 0, 3] @Int)
