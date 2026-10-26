{-# LANGUAGE DataKinds #-}

module TransposeScalar where

import qualified Synapse.Tensor as T

bad = T.transpose (T.scalar (T.element (1 :: Int)))
