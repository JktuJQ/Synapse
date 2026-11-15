{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module BackwardCapability where

import Data.Proxy (Proxy (..))
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import Synapse.Tensor.Ops (TensorOps (..))

class ReverseCapability a where
    reverseCotangent :: T.Tensor '[1] a -> T.Tensor '[1] a

operation
    :: (TensorOps t Double, BackwardConstraints t (ReverseCapability Double))
    => t '[1] Double
    -> t '[1] Double
operation = customVJP (Proxy @(ReverseCapability Double)) id (,()) (\() -> reverseCotangent)

-- Ordinary evaluation needs no instance, but differentiation must reject its absence.
ordinary :: T.Tensor '[1] Double
ordinary = operation 1

bad :: T.Tensor '[1] Double
bad = AG.gradient operation ordinary
