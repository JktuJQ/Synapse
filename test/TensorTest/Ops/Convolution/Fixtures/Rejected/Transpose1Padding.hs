{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Transpose1Padding where

import qualified Synapse.Tensor as T

bad = T.convTranspose1d @'[2] @'[0, 0] @'[1] @1 @'[2] (T.ones @'[1, 1, 2] @Double) (T.ones @'[1, 1, 2])
