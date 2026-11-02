{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Transpose1Stride where

import qualified Synapse.Tensor as T

bad = T.convTranspose1d @'[0] @'[0, 0] @'[1] @1 @'[0] (T.ones @'[1, 1, 2] @Double) (T.ones @'[1, 1, 2])
