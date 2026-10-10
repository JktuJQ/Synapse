{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SliceStopBounds where

import qualified Synapse.Tensor as T

bad = T.slice @'[ 'T.Range 1 5 1] (T.zeros @'[4] :: T.Tensor '[4] Int)
