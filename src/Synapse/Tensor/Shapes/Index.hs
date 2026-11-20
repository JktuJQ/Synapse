{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Type-level coordinates, slices, and shape constraints for indexed selection.
module Synapse.Tensor.Shapes.Index
    ( -- * Static coordinates
      Index (..)
    , CanIndex

      -- * Static slices
    , SliceAxis (..)
    , Slice (..)
    , CanSlice
    , SliceShape

      -- * Tensor indices
    , CanGather
    , LookupRowsShape
    )
where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeLits
    ( CmpNat
    , Div
    , ErrorMessage (ShowType, Text, (:<>:))
    , KnownNat
    , Nat
    , TypeError
    , natVal
    , type (+)
    , type (-)
    )

import Synapse.Tensor.Shapes.Axes (CanSelectAxes, RemoveAxis)

-- Selection specifications and runtime values

{- | Selection for one input axis.
Ranges have an exclusive stop and a positive step magnitude.
'RangeReverse' reverses the selected range; it does not use negative coordinates or a negative step.
-}
data SliceAxis a
    = -- | Keep the entire axis.
      All
    | -- | Select a zero-based coordinate and remove the axis from the result.
      At a
    | {- | Select coordinates from @start@ up to exclusive @stop@, separated by @step@.
      A stop at or before the start produces an empty axis.
      -}
      Range a a a
    | -- | Select the same coordinates as 'Range' and return them in reverse order.
      RangeReverse a a a
    deriving stock (Show)

-- | Reflect a static coordinate or range parameter as Int; its value must fit in that type.
natInt
    :: forall n
     . KnownNat n
    => Int
natInt = fromInteger (toInteger (natVal (Proxy @n)))

-- Static coordinates

-- | Require a coordinate strictly below its dimension size and report the axis on failure.
type family CheckCoordinate (order :: Ordering) (axis :: Nat) (coordinate :: Nat) (dim :: Nat) :: Constraint where
    CheckCoordinate 'LT axis coordinate dim = ()
    CheckCoordinate order axis coordinate dim =
        TypeError
            ( 'Text "Index "
                ':<>: 'ShowType coordinate
                ':<>: 'Text " is out of bounds for axis "
                ':<>: 'ShowType axis
                ':<>: 'Text " of size "
                ':<>: 'ShowType dim
            )

-- | Check coordinate count and bounds while advancing through the input axes.
type family CheckIndex (axis :: Nat) (coordinates :: [Nat]) (sh :: [Nat]) :: Constraint where
    CheckIndex axis '[] '[] = ()
    CheckIndex axis '[] (dim ': rest) =
        TypeError ('Text "Index has too few coordinates: missing axis " ':<>: 'ShowType axis)
    CheckIndex axis (coordinate ': rest) '[] =
        TypeError ('Text "Index has too many coordinates: unexpected axis " ':<>: 'ShowType axis)
    CheckIndex axis (coordinate ': coordinates) (dim ': rest) =
        (CheckCoordinate (CmpNat coordinate dim) axis coordinate dim, CheckIndex (axis + 1) coordinates rest)

{- | Require exactly one in-bounds coordinate per input dimension.
The empty coordinate list indexes a scalar.
No coordinate is valid for a zero-sized dimension.
-}
type CanIndex coordinates sh = CheckIndex 0 coordinates sh

{- | Reflect a static coordinate list in left-to-right axis order.
'CanIndex' separately checks its length and bounds against an input shape.
-}
class Index (coordinates :: [Nat]) where
    -- | Runtime coordinates in the same order as the type-level list.
    coordinates
        :: [Int]

instance Index '[] where
    coordinates = []
instance (KnownNat coordinate, Index rest) => Index (coordinate ': rest) where
    coordinates = natInt @coordinate : coordinates @rest

-- Static slices

-- | Require a slice boundary not to exceed the dimension size; equality is permitted.
type family CheckBoundary (order :: Ordering) (axis :: Nat) (boundary :: Nat) (dim :: Nat) :: Constraint where
    CheckBoundary 'GT axis boundary dim =
        TypeError
            ( 'Text "Slice boundary "
                ':<>: 'ShowType boundary
                ':<>: 'Text " is out of bounds for axis "
                ':<>: 'ShowType axis
                ':<>: 'Text " of size "
                ':<>: 'ShowType dim
            )
    CheckBoundary order axis boundary dim = ()

-- | Reject a zero slice step.
type family CheckStep (step :: Nat) :: Constraint where
    CheckStep 0 = TypeError ('Text "Slice step must be positive")
    CheckStep step = ()

-- | Compute a positive range length with ceiling division, or zero when the stop precedes the start.
type family RangeLengthOrdered (order :: Ordering) (start :: Nat) (stop :: Nat) (step :: Nat) :: Nat where
    RangeLengthOrdered 'LT start stop step = Div ((stop - start) + (step - 1)) step
    RangeLengthOrdered order start stop step = 0

-- | Compute the number of selected coordinates, rejecting a zero step.
type family RangeLength (start :: Nat) (stop :: Nat) (step :: Nat) :: Nat where
    RangeLength start stop 0 = TypeError ('Text "Slice step must be positive")
    RangeLength start stop step = RangeLengthOrdered (CmpNat start stop) start stop step

-- | Validate both slice boundaries and require a positive step.
type CheckRange axis start stop step dim =
    ( CheckBoundary (CmpNat start dim) axis start dim
    , CheckBoundary (CmpNat stop dim) axis stop dim
    , CheckStep step
    )

-- | Validate one selector against its input dimension.
type family CheckSliceAxis (axis :: Nat) (selection :: SliceAxis Nat) (dim :: Nat) :: Constraint where
    CheckSliceAxis axis 'All dim = ()
    CheckSliceAxis axis ('At coordinate) dim = CheckCoordinate (CmpNat coordinate dim) axis coordinate dim
    CheckSliceAxis axis ('Range start stop step) dim = CheckRange axis start stop step dim
    CheckSliceAxis axis ('RangeReverse start stop step) dim = CheckRange axis start stop step dim

-- | Check selection count and bounds while advancing through the input axes.
type family CheckSlice (axis :: Nat) (selections :: [SliceAxis Nat]) (sh :: [Nat]) :: Constraint where
    CheckSlice axis '[] '[] = ()
    CheckSlice axis '[] (dim ': rest) =
        TypeError ('Text "Slice has too few selections: missing axis " ':<>: 'ShowType axis)
    CheckSlice axis (selection ': rest) '[] =
        TypeError ('Text "Slice has too many selections: unexpected axis " ':<>: 'ShowType axis)
    CheckSlice axis (selection ': selections) (dim ': rest) =
        (CheckSliceAxis axis selection dim, CheckSlice (axis + 1) selections rest)

{- | Require exactly one valid selection per input dimension.

Coordinates selected by 'At' must lie strictly inside the axis.
Range boundaries may equal the dimension size, but may not exceed it; steps must be positive.
A range whose stop is at or before its start produces an empty axis.
The empty selection list is valid for a scalar.
-}
type CanSlice selections sh = CheckSlice 0 selections sh

{- | Shape of a statically selected region.

'At' removes an axis; 'All' and ranges retain it.
Reversing a range does not change its output length.
The selection must describe every input axis; bounds are checked separately by 'CanSlice'.

@
SliceShape '['All, 'At 1] '[2, 3] = '[2]
SliceShape '['Range 0 5 2] '[5]   = '[3]
@
-}
type family SliceShape (spec :: [SliceAxis Nat]) (sh :: [Nat]) :: [Nat] where
    SliceShape '[] '[] = '[]
    SliceShape ('All ': spec) (dim ': rest) = dim ': SliceShape spec rest
    SliceShape ('At coordinate ': spec) (dim ': rest) = SliceShape spec rest
    SliceShape ('Range start stop step ': spec) (dim ': rest) =
        RangeLength start stop step ': SliceShape spec rest
    SliceShape ('RangeReverse start stop step ': spec) (dim ': rest) =
        RangeLength start stop step ': SliceShape spec rest
    SliceShape spec sh = TypeError ('Text "Slice must describe every tensor axis")

{- | Reflect a static per-axis selection list in left-to-right order.
'CanSlice' separately checks the selections against an input shape.
-}
class Slice (spec :: [SliceAxis Nat]) where
    -- | Runtime selectors in the same order as the type-level list.
    sliceAxes
        :: [SliceAxis Int]

instance Slice '[] where
    sliceAxes = []
instance Slice rest => Slice ('All ': rest) where
    sliceAxes = All : sliceAxes @rest
instance (KnownNat coordinate, Slice rest) => Slice ('At coordinate ': rest) where
    sliceAxes = At (natInt @coordinate) : sliceAxes @rest
instance (KnownNat start, KnownNat stop, KnownNat step, Slice rest) => Slice ('Range start stop step ': rest) where
    sliceAxes = Range (natInt @start) (natInt @stop) (natInt @step) : sliceAxes @rest
instance (KnownNat start, KnownNat stop, KnownNat step, Slice rest) => Slice ('RangeReverse start stop step ': rest) where
    sliceAxes = RangeReverse (natInt @start) (natInt @stop) (natInt @step) : sliceAxes @rest

-- Tensor indices

{- | Require equal source and index ranks, with matching dimensions outside the selected axis.

The selected axis must exist in both shapes; its size may differ between them.
The gathered result has the index tensor's shape.
This constraint does not establish that runtime index values are within source bounds.
-}
type CanGather axis source indices =
    ( CanSelectAxes '[axis] source
    , CanSelectAxes '[axis] indices
    , RemoveAxis axis source ~ RemoveAxis axis indices
    )

{- | Shape of complete rows selected by an arbitrary-shaped index tensor.
Append the row width to the index shape: @LookupRowsShape '[2, 3] 4@ is @'[2, 3, 4]@.
A scalar index selects one row with shape @'[width]@.
-}
type family LookupRowsShape (indices :: [Nat]) (width :: Nat) :: [Nat] where
    LookupRowsShape '[] width = '[width]
    LookupRowsShape (dim ': rest) width = dim ': LookupRowsShape rest width
