{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module IndexBounds where

import qualified Synapse.Tensor as T

bad = T.index @'[1, 3] (T.zeros @'[2, 3] :: T.Tensor '[2, 3] Int)
