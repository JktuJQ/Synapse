{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Pool3Padding where

import qualified Synapse.Tensor as T

bad = T.avgPool3d @'[2, 2, 2] @'[1, 1, 1] @'[0, 0, 2, 0, 0, 0] False (T.ones @'[1, 1, 2, 2, 2] @Double)
