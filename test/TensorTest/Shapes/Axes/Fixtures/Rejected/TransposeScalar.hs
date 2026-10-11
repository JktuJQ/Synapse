{-# LANGUAGE DataKinds #-}

module TransposeScalar where

import qualified Synapse.Tensor as T

bad = T.transpose (T.scalar (T.constant (1 :: Int)))
