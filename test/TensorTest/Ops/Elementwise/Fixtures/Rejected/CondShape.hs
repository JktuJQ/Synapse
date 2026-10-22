{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module CondShape where

import qualified Synapse.Tensor as T

bad = T.cond (T.scalar (T.constant True)) (T.zeros @'[2] @Int) (T.ones @'[3])
