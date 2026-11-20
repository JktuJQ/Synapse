{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Reduction axes for layer normalization and batch normalization.
module Synapse.Tensor.Shapes.Normalization (LayerNormAxes, BatchNormAxes) where

import GHC.TypeLits (ErrorMessage (..), Nat, TypeError, type (+), type (-))

import Synapse.Tensor.Shapes.Axes (Rank)
import Synapse.Tensor.Shapes.Backend (Reverse)

-- Layer normalization

-- | Construct a list of consecutive axes beginning at the supplied position.
type family AxisRange (start :: Nat) (count :: Nat) :: [Nat] where
    AxisRange start 0 = '[]
    AxisRange start count = start ': AxisRange (start + 1) (count - 1)

-- | Match reversed normalized dimensions against reversed input dimensions before returning the result axes.
type family CheckSuffix (normalized :: [Nat]) (input :: [Nat]) (result :: [Nat]) :: [Nat] where
    CheckSuffix '[] input result = result
    CheckSuffix (dim ': rest) (dim ': input) result = CheckSuffix rest input result
    CheckSuffix normalized input result =
        TypeError ('Text "LayerNorm dimensions must match the trailing input dimensions")

{- | Select trailing input axes whose dimensions exactly match the normalized shape.

The normalized shape must be nonempty and match a suffix of the input shape.
@LayerNormAxes '[3, 4] '[2, 3, 4]@ is @'[1, 2]@.
A mismatched suffix or an input of insufficient rank produces a type error.
-}
type family LayerNormAxes (normalized :: [Nat]) (sh :: [Nat]) :: [Nat] where
    LayerNormAxes '[] sh = TypeError ('Text "LayerNorm requires at least one normalized dimension")
    LayerNormAxes normalized sh =
        CheckSuffix (Reverse normalized) (Reverse sh) (AxisRange (Rank sh - Rank normalized) (Rank normalized))

-- Batch normalization

-- | Enumerate a consecutive range of axes, omitting the excluded axis if present.
type family WithoutAxis (excluded :: Nat) (current :: Nat) (remaining :: Nat) :: [Nat] where
    WithoutAxis excluded current 0 = '[]
    WithoutAxis axis axis remaining = WithoutAxis axis (axis + 1) (remaining - 1)
    WithoutAxis excluded current remaining = current ': WithoutAxis excluded (current + 1) (remaining - 1)

{- | Select every input axis except the channel axis, retaining their original order.

@BatchNormAxes 1 '[8, 16, 32, 32]@ is @'[0, 2, 3]@.
This family does not check that the channel axis exists; callers must validate it separately.
If it is outside the input rank, every input axis is returned.
-}
type BatchNormAxes channel sh = WithoutAxis channel 0 (Rank sh)
