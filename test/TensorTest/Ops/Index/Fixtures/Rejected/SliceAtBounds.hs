{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SliceAtBounds where

import qualified Synapse.Tensor as T

bad = T.slice @'[ 'T.All, 'T.At 3] (T.zeros @'[2, 3] :: T.Tensor '[2, 3] Int)
