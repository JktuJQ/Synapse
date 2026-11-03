{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Static coordinates and per-axis selections.
-- Every axis must be described.
module Synapse.Tensor.Shapes.Index
  ( SliceAxis (..),
    Index (..),
    Slice (..),
    CanIndex,
    CanGather,
    LookupRowsShape,
    CanSlice,
    SliceShape,
  )
where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeLits
  ( CmpNat,
    Div,
    ErrorMessage (ShowType, Text, (:<>:)),
    KnownNat,
    Nat,
    TypeError,
    natVal,
    type (+),
    type (-),
  )
import Synapse.Tensor.Shapes.Axes (CanSelectAxes, RemoveAxis)

-- | Gather indices have the same rank and dimensions as the source outside the selected axis.
-- The result has the shape of the indices.
type CanGather axis source indices =
  (CanSelectAxes '[axis] source, CanSelectAxes '[axis] indices, RemoveAxis axis source ~ RemoveAxis axis indices)

-- | Shape of complete rows selected by an arbitrary-shaped index tensor.
-- Appends the row width to the index shape, so @'[batch, time]@ becomes @'[batch, time, width]@.
type family LookupRowsShape (indices :: [Nat]) (width :: Nat) :: [Nat] where
  LookupRowsShape '[] width = '[width]
  LookupRowsShape (dim ': rest) width = dim ': LookupRowsShape rest width

-- | Ranges use an exclusive stop and a positive step.
data SliceAxis a
  = -- | Keep the entire axis.
    All
  | -- | Select a zero-based coordinate and remove the axis from the result.
    At a
  | -- | @Range start stop step@ selects coordinates starting at @start@, strictly below @stop@, spaced by @step@.
    -- A stop at or before the start produces an empty axis.
    Range a a a
  | -- | @RangeReverse start stop step@ selects the same coordinates as 'Range' and returns them in reverse order.
    RangeReverse a a a

type family CheckCoordinate (order :: Ordering) (axis :: Nat) (coordinate :: Nat) (dim :: Nat) :: Constraint where
  CheckCoordinate 'LT axis coordinate dim = ()
  CheckCoordinate order axis coordinate dim =
    TypeError ('Text "Index " ':<>: 'ShowType coordinate ':<>: 'Text " is out of bounds for axis " ':<>: 'ShowType axis ':<>: 'Text " of size " ':<>: 'ShowType dim)

type family CheckIndex (axis :: Nat) (coordinates :: [Nat]) (sh :: [Nat]) :: Constraint where
  CheckIndex axis '[] '[] = ()
  CheckIndex axis '[] (dim ': rest) =
    TypeError ('Text "Index has too few coordinates: missing axis " ':<>: 'ShowType axis)
  CheckIndex axis (coordinate ': rest) '[] =
    TypeError ('Text "Index has too many coordinates: unexpected axis " ':<>: 'ShowType axis)
  CheckIndex axis (coordinate ': coordinates) (dim ': rest) =
    (CheckCoordinate (CmpNat coordinate dim) axis coordinate dim, CheckIndex (axis + 1) coordinates rest)

-- | Exactly one in-bounds coordinate per dimension, including none for a scalar.
type CanIndex coordinates sh = CheckIndex 0 coordinates sh

type family CheckBoundary (order :: Ordering) (axis :: Nat) (boundary :: Nat) (dim :: Nat) :: Constraint where
  CheckBoundary 'GT axis boundary dim =
    TypeError ('Text "Slice boundary " ':<>: 'ShowType boundary ':<>: 'Text " is out of bounds for axis " ':<>: 'ShowType axis ':<>: 'Text " of size " ':<>: 'ShowType dim)
  CheckBoundary order axis boundary dim = ()

type family CheckStep (step :: Nat) :: Constraint where
  CheckStep 0 = TypeError ('Text "Slice step must be positive")
  CheckStep step = ()

type family RangeLengthOrdered (order :: Ordering) (start :: Nat) (stop :: Nat) (step :: Nat) :: Nat where
  RangeLengthOrdered 'LT start stop step = Div ((stop - start) + (step - 1)) step
  RangeLengthOrdered order start stop step = 0

type family RangeLength (start :: Nat) (stop :: Nat) (step :: Nat) :: Nat where
  RangeLength start stop 0 = TypeError ('Text "Slice step must be positive")
  RangeLength start stop step = RangeLengthOrdered (CmpNat start stop) start stop step

type CheckRange axis start stop step dim =
  (CheckBoundary (CmpNat start dim) axis start dim, CheckBoundary (CmpNat stop dim) axis stop dim, CheckStep step)

type family CheckSliceAxis (axis :: Nat) (selection :: SliceAxis Nat) (dim :: Nat) :: Constraint where
  CheckSliceAxis axis 'All dim = ()
  CheckSliceAxis axis ('At coordinate) dim = CheckCoordinate (CmpNat coordinate dim) axis coordinate dim
  CheckSliceAxis axis ('Range start stop step) dim = CheckRange axis start stop step dim
  CheckSliceAxis axis ('RangeReverse start stop step) dim = CheckRange axis start stop step dim

type family CheckSlice (axis :: Nat) (selections :: [SliceAxis Nat]) (sh :: [Nat]) :: Constraint where
  CheckSlice axis '[] '[] = ()
  CheckSlice axis '[] (dim ': rest) =
    TypeError ('Text "Slice has too few selections: missing axis " ':<>: 'ShowType axis)
  CheckSlice axis (selection ': rest) '[] =
    TypeError ('Text "Slice has too many selections: unexpected axis " ':<>: 'ShowType axis)
  CheckSlice axis (selection ': selections) (dim ': rest) =
    (CheckSliceAxis axis selection dim, CheckSlice (axis + 1) selections rest)

-- | Exactly one selection per dimension, with strict bounds and nonzero strides.
type CanSlice selections sh = CheckSlice 0 selections sh

-- | Shape of a selected region.
-- At removes an axis; other selectors retain it.
-- Bounds are checked separately by CanSlice.
type family SliceShape (spec :: [SliceAxis Nat]) (sh :: [Nat]) :: [Nat] where
  SliceShape '[] '[] = '[]
  SliceShape ('All ': spec) (dim ': rest) = dim ': SliceShape spec rest
  SliceShape ('At coordinate ': spec) (dim ': rest) = SliceShape spec rest
  SliceShape ('Range start stop step ': spec) (dim ': rest) = RangeLength start stop step ': SliceShape spec rest
  SliceShape ('RangeReverse start stop step ': spec) (dim ': rest) = RangeLength start stop step ': SliceShape spec rest
  SliceShape spec sh = TypeError ('Text "Slice must describe every tensor axis")

natInt :: forall n. (KnownNat n) => Int
natInt = fromInteger (toInteger (natVal (Proxy @n)))

-- | Materialize a static coordinate list; CanIndex checks it against a shape.
class Index (coordinates :: [Nat]) where
  coordinates :: [Int]

instance Index '[] where
  coordinates = []

instance (KnownNat coordinate, Index rest) => Index (coordinate ': rest) where
  coordinates = natInt @coordinate : coordinates @rest

-- | Materialize static selectors; CanSlice checks their applicability.
class Slice (spec :: [SliceAxis Nat]) where
  sliceAxes :: [SliceAxis Int]

instance Slice '[] where
  sliceAxes = []

instance (Slice rest) => Slice ('All ': rest) where
  sliceAxes = All : sliceAxes @rest

instance (KnownNat coordinate, Slice rest) => Slice ('At coordinate ': rest) where
  sliceAxes = At (natInt @coordinate) : sliceAxes @rest

instance (KnownNat start, KnownNat stop, KnownNat step, Slice rest) => Slice ('Range start stop step ': rest) where
  sliceAxes = Range (natInt @start) (natInt @stop) (natInt @step) : sliceAxes @rest

instance (KnownNat start, KnownNat stop, KnownNat step, Slice rest) => Slice ('RangeReverse start stop step ': rest) where
  sliceAxes = RangeReverse (natInt @start) (natInt @stop) (natInt @step) : sliceAxes @rest
