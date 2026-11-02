{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Axes selected by normalization operations.
module Synapse.Tensor.Shapes.Normalization (LayerNormAxes, BatchNormAxes) where

import GHC.TypeLits (ErrorMessage (..), Nat, TypeError, type (+), type (-))
import Synapse.Tensor.Shapes.Axes (Rank)
import Synapse.Tensor.Shapes.Dimensions (Reverse)

type family AxisRange (start :: Nat) (count :: Nat) :: [Nat] where
  AxisRange start 0 = '[]
  AxisRange start count = start ': AxisRange (start + 1) (count - 1)

type family CheckSuffix (normalized :: [Nat]) (input :: [Nat]) (result :: [Nat]) :: [Nat] where
  CheckSuffix '[] input result = result
  CheckSuffix (dim ': rest) (dim ': input) result = CheckSuffix rest input result
  CheckSuffix normalized input result = TypeError ('Text "LayerNorm dimensions must match the trailing input dimensions")

-- | Select trailing axes whose dimensions match the normalized shape.
type family LayerNormAxes (normalized :: [Nat]) (sh :: [Nat]) :: [Nat] where
  LayerNormAxes '[] sh = TypeError ('Text "LayerNorm requires at least one normalized dimension")
  LayerNormAxes normalized sh = CheckSuffix (Reverse normalized) (Reverse sh) (AxisRange (Rank sh - Rank normalized) (Rank normalized))

type family WithoutAxis (excluded :: Nat) (current :: Nat) (remaining :: Nat) :: [Nat] where
  WithoutAxis excluded current 0 = '[]
  WithoutAxis axis axis remaining = WithoutAxis axis (axis + 1) (remaining - 1)
  WithoutAxis excluded current remaining = current ': WithoutAxis excluded (current + 1) (remaining - 1)

-- | Select every axis except the channel axis.
type BatchNormAxes channel sh = WithoutAxis channel 0 (Rank sh)
