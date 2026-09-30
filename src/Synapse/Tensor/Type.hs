{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

-- | Core tensor type.
module Synapse.Tensor.Type
  ( Tensor (..),
  )
where

import qualified Data.Array.Accelerate as A
import GHC.TypeLits (Nat)
import Synapse.Tensor.Shape (ShapeToDIM)

-- | A staged tensor expression.
newtype Tensor (sh :: [Nat]) a
  = Tensor {unTensor :: A.Acc (A.Array (ShapeToDIM sh) a)}
