{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Transpose3Size where

import qualified Synapse.Tensor as T

bad = T.convTranspose3d @'[1, 1, 1] @'[0, 0, 2, 2, 0, 0] @'[1, 1, 1] @1 @'[0, 0, 0] (T.ones @'[1, 1, 2, 2, 2] @Double) (T.ones @'[1, 1, 1, 1, 1])
