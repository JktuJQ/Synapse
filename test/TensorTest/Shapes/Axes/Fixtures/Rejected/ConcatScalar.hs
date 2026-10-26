{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ConcatScalar where

import qualified Synapse.Tensor as T

bad = T.concat @0 (T.scalar (T.element (1 :: Int))) (T.scalar 2)
