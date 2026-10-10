{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SliceZeroStep where

import qualified Synapse.Tensor as T

bad = T.slice @'[ 'T.Range 0 4 0] (T.zeros @'[4] :: T.Tensor '[4] Int)
