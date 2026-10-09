{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Axes are numbered from zero, from left to right. Axis lists retain their
-- order and may not contain duplicates. Insertion also permits the final
-- position, immediately after the last axis.
module Synapse.Tensor.Shapes.Axes
  ( Rank,
    AxisSize,
    RemoveAxis,
    InsertAxis,
    SelectAxes,
    PermuteAxes,
    CanSelectAxes,
    CanPermuteAxes,
    CanSqueeze,
    Axes (..),
  )
where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeLits
  ( CmpNat,
    ErrorMessage (ShowType, Text, (:<>:)),
    KnownNat,
    Nat,
    TypeError,
    natVal,
    type (+),
    type (-),
  )

-- | Number of dimensions (or entries in an axis list).
type family Rank (sh :: [Nat]) :: Nat where
  Rank '[] = 0
  Rank (_ ': rest) = 1 + Rank rest

type family LookupAxisAt (axis :: Nat) (sh :: [Nat]) :: Nat where
  LookupAxisAt 0 (dim ': rest) = dim
  LookupAxisAt axis (dim ': rest) = LookupAxisAt (axis - 1) rest

type family LookupAxis (order :: Ordering) (axis :: Nat) (sh :: [Nat]) :: Nat where
  LookupAxis 'LT axis sh = LookupAxisAt axis sh
  LookupAxis order axis sh =
    TypeError ('Text "Axis " ':<>: 'ShowType axis ':<>: 'Text " is out of bounds for shape " ':<>: 'ShowType sh)

-- | Size of an existing axis. Scalars have no existing axes.
type AxisSize axis sh = LookupAxis (CmpNat axis (Rank sh)) axis sh

type family RemoveAxisAt (axis :: Nat) (sh :: [Nat]) :: [Nat] where
  RemoveAxisAt 0 (dim ': rest) = rest
  RemoveAxisAt axis (dim ': rest) = dim ': RemoveAxisAt (axis - 1) rest

type family RemoveAxisChecked (order :: Ordering) (axis :: Nat) (sh :: [Nat]) :: [Nat] where
  RemoveAxisChecked 'LT axis sh = RemoveAxisAt axis sh
  RemoveAxisChecked order axis sh =
    TypeError ('Text "Cannot remove axis " ':<>: 'ShowType axis ':<>: 'Text " from shape " ':<>: 'ShowType sh)

-- | Remove an existing dimension, regardless of its size.
-- This is a shape transformation, not a value-preserving tensor operation.
type RemoveAxis axis sh = RemoveAxisChecked (CmpNat axis (Rank sh)) axis sh

type family InsertAxisAt (axis :: Nat) (dim :: Nat) (sh :: [Nat]) :: [Nat] where
  InsertAxisAt 0 dim sh = dim ': sh
  InsertAxisAt axis dim (head ': rest) = head ': InsertAxisAt (axis - 1) dim rest

type family InsertAxisChecked (order :: Ordering) (axis :: Nat) (dim :: Nat) (sh :: [Nat]) :: [Nat] where
  InsertAxisChecked 'GT axis dim sh =
    TypeError ('Text "Cannot insert axis " ':<>: 'ShowType axis ':<>: 'Text " into shape " ':<>: 'ShowType sh)
  InsertAxisChecked order axis dim sh = InsertAxisAt axis dim sh

-- | Insert a dimension at a position from zero through the rank, inclusive.
type InsertAxis axis dim sh = InsertAxisChecked (CmpNat axis (Rank sh)) axis dim sh

type family Contains (axis :: Nat) (xs :: [Nat]) :: Bool where
  Contains axis '[] = 'False
  Contains axis (axis ': rest) = 'True
  Contains axis (other ': rest) = Contains axis rest

type family SelectAxis (duplicate :: Bool) (axis :: Nat) (rest :: [Nat]) (sh :: [Nat]) :: [Nat] where
  SelectAxis 'True axis rest sh =
    TypeError ('Text "Repeated axis " ':<>: 'ShowType axis ':<>: 'Text " in axis selection")
  SelectAxis 'False axis rest sh = AxisSize axis sh ': SelectAxes rest sh

-- | Select dimensions in the requested order, with no repeated axes.
type family SelectAxes (xs :: [Nat]) (sh :: [Nat]) :: [Nat] where
  SelectAxes '[] sh = '[]
  SelectAxes (axis ': rest) sh = SelectAxis (Contains axis rest) axis rest sh

type family PermuteAxesChecked (order :: Ordering) (xs :: [Nat]) (sh :: [Nat]) :: [Nat] where
  PermuteAxesChecked 'EQ xs sh = SelectAxes xs sh
  PermuteAxesChecked order xs sh =
    TypeError ('Text "Axis permutation " ':<>: 'ShowType xs ':<>: 'Text " must contain every axis of shape " ':<>: 'ShowType sh)

-- | Reorder all dimensions. Every source axis must occur exactly once.
type PermuteAxes xs sh = PermuteAxesChecked (CmpNat (Rank xs) (Rank sh)) xs sh

type family CheckAxis (order :: Ordering) (axis :: Nat) (sh :: [Nat]) :: Constraint where
  CheckAxis 'LT axis sh = ()
  CheckAxis order axis sh =
    TypeError ('Text "Axis " ':<>: 'ShowType axis ':<>: 'Text " is out of bounds for shape " ':<>: 'ShowType sh)

type family CheckDuplicate (duplicate :: Bool) (axis :: Nat) :: Constraint where
  CheckDuplicate 'False axis = ()
  CheckDuplicate 'True axis =
    TypeError ('Text "Repeated axis " ':<>: 'ShowType axis ':<>: 'Text " in axis selection")

-- | Check a possibly partial selection of unique existing axes.
type family CanSelectAxes (xs :: [Nat]) (sh :: [Nat]) :: Constraint where
  CanSelectAxes '[] sh = ()
  CanSelectAxes (axis ': rest) sh =
    (CheckAxis (CmpNat axis (Rank sh)) axis sh, CheckDuplicate (Contains axis rest) axis, CanSelectAxes rest sh)

type family CheckPermutationLength (order :: Ordering) (xs :: [Nat]) (sh :: [Nat]) :: Constraint where
  CheckPermutationLength 'EQ xs sh = ()
  CheckPermutationLength order xs sh =
    TypeError ('Text "Axis permutation " ':<>: 'ShowType xs ':<>: 'Text " must contain every axis of shape " ':<>: 'ShowType sh)

-- | Check a complete permutation, including its length, bounds, and uniqueness.
type CanPermuteAxes xs sh = (CanSelectAxes xs sh, CheckPermutationLength (CmpNat (Rank xs) (Rank sh)) xs sh)

type family CheckSingleton (axis :: Nat) (sh :: [Nat]) (dim :: Nat) :: Constraint where
  CheckSingleton axis sh 1 = ()
  CheckSingleton axis sh dim =
    TypeError ('Text "Cannot squeeze axis " ':<>: 'ShowType axis ':<>: 'Text " of shape " ':<>: 'ShowType sh ':<>: 'Text ": expected size 1, got " ':<>: 'ShowType dim)

-- | Only a singleton dimension can be squeezed without removing elements.
type CanSqueeze axis sh = CheckSingleton axis sh (AxisSize axis sh)

-- | Materialize an axis list while preserving its order.
class Axes (xs :: [Nat]) where
  axes :: [Int]

instance Axes '[] where
  axes = []

instance (KnownNat axis, Axes rest) => Axes (axis ': rest) where
  axes = fromInteger (toInteger (natVal (Proxy @axis))) : axes @rest
