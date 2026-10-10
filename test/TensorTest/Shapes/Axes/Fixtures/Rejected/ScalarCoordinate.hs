{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ScalarCoordinate where

import qualified Synapse.Tensor as T

bad = T.fromGen @'[] (T.coordinate @0)
