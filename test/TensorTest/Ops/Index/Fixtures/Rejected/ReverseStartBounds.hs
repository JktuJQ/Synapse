{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ReverseStartBounds where

import qualified Synapse.Tensor as T

bad = T.slice @'[ 'T.RangeReverse 5 2 1] (T.zeros @'[4] :: T.Tensor '[4] Int)
