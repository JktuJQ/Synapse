{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SliceStartBounds where

import qualified Synapse.Tensor as T

bad = T.slice @'[ 'T.Range 5 2 1] (T.zeros @'[4] :: T.Tensor '[4] Int)
