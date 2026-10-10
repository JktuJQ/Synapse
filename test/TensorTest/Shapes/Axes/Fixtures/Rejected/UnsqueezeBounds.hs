{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module UnsqueezeBounds where

import qualified Synapse.Tensor as T

bad = T.unsqueeze @3 (T.zeros @'[2, 3] :: T.Tensor '[2, 3] Int)
