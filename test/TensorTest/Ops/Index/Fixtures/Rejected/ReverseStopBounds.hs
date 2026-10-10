{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ReverseStopBounds where

import qualified Synapse.Tensor as T

bad = T.slice @'[ 'T.RangeReverse 1 5 1] (T.zeros @'[4] :: T.Tensor '[4] Int)
