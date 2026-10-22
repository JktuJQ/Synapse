{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module IntegerMean where

import qualified Synapse.Tensor as T

bad :: T.Tensor '[] Int
bad = T.mean (T.ones @'[3] @Int)
