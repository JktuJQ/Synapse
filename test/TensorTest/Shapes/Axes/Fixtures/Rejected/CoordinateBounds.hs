{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module CoordinateBounds where

import qualified Synapse.Tensor as T

bad = T.fromGen @'[2, 3] (T.coordinate @2)
