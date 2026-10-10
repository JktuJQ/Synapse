{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SliceLong where

import qualified Synapse.Tensor as T

bad = T.slice @'[ 'T.All, 'T.All, 'T.All] (T.zeros @'[2, 3] :: T.Tensor '[2, 3] Int)
