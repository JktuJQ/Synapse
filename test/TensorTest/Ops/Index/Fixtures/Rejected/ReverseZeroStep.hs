{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ReverseZeroStep where

import qualified Synapse.Tensor as T

bad = T.slice @'[ 'T.RangeReverse 0 4 0] (T.zeros @'[4] :: T.Tensor '[4] Int)
