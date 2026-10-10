{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module IndexEmpty where

import qualified Synapse.Tensor as T

bad = T.index @'[0] (T.zeros @'[0] :: T.Tensor '[0] Int)
