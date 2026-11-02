{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TransposeSize where

import qualified Synapse.Tensor as T

bad = T.convTranspose2d @'[1, 1] @'[3, 3, 0, 0] @'[1, 1] @1 @'[0, 0] (T.ones @'[1, 1, 2, 2] @Double) (T.ones @'[1, 1, 2, 2])
