{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Transpose3Channels where

import qualified Synapse.Tensor as T

bad = T.convTranspose3d @'[1, 1, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1] @1 @'[0, 0, 0] (T.ones @'[1, 2, 2, 2, 2] @Double) (T.ones @'[3, 1, 1, 1, 1])
