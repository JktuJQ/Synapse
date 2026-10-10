{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module MissingAxis where

import qualified Synapse.Tensor as T

bad = T.sumAxes @'[2] (T.zeros @'[2, 3] @Int)
