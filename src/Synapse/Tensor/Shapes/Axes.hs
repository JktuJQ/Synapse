{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Type-level axis selection, permutation, insertion, and reduction.
module Synapse.Tensor.Shapes.Axes
    ( -- * Axis witnesses and dimensions
      Axes (..)
    , Rank
    , AxisSize

      -- * Axis transformations
    , RemoveAxis
    , InsertAxis
    , SelectAxes
    , CanSelectAxes
    , CanSqueeze
    , PermuteAxes
    , TransposeShape
    , ConcatShape

      -- * Reduction
    , ReduceShape
    , CanReduce
    , CanReduceNonEmpty
    , NonEmpty
    )
where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeLits
    ( CmpNat
    , ErrorMessage (ShowType, Text, (:<>:))
    , KnownNat
    , Nat
    , TypeError
    , natVal
    , type (+)
    , type (-)
    )

import Synapse.Tensor.Shapes.Backend (Reverse)
import Synapse.Tensor.Shapes.Dimensions (ShapeSize)

-- Axis lists and witnesses

{- | Number of dimensions, or entries in an axis list.
@Rank '[]@ is @0@ and @Rank '[2, 3, 4]@ is @3@.
-}
type family Rank (sh :: [Nat]) :: Nat where
    Rank '[] = 0
    Rank (_ ': rest) = 1 + Rank rest

-- | Test whether an axis occurs in a type-level list.
type family Contains (axis :: Nat) (xs :: [Nat]) :: Bool where
    Contains axis '[] = 'False
    Contains axis (axis ': rest) = 'True
    Contains axis (other ': rest) = Contains axis rest

{- | Reflect a type-level axis list in its declared order.
This class does not check bounds or uniqueness; use the constraint required by the tensor operation.
-}
class Axes (xs :: [Nat]) where
    -- | Runtime axis numbers in the declared order.
    axes
        :: [Int]

instance Axes '[] where
    axes = []
instance (KnownNat axis, Axes rest) => Axes (axis ': rest) where
    axes = fromInteger (toInteger (natVal (Proxy @axis))) : axes @rest

-- Dimension lookup

-- | Look up a dimension after the caller has established that the axis exists.
type family LookupAxisAt (axis :: Nat) (sh :: [Nat]) :: Nat where
    LookupAxisAt 0 (dim ': rest) = dim
    LookupAxisAt axis (dim ': rest) = LookupAxisAt (axis - 1) rest

-- | Return the selected dimension or report an out-of-bounds axis.
type family LookupAxis (order :: Ordering) (axis :: Nat) (sh :: [Nat]) :: Nat where
    LookupAxis 'LT axis sh = LookupAxisAt axis sh
    LookupAxis order axis sh =
        TypeError ('Text "Axis " ':<>: 'ShowType axis ':<>: 'Text " is out of bounds for shape " ':<>: 'ShowType sh)

{- | Size of an existing zero-based axis.
@AxisSize 1 '[2, 3, 4]@ is @3@.
Scalars have no axes; an out-of-bounds selection produces a type error.
-}
type AxisSize axis sh = LookupAxis (CmpNat axis (Rank sh)) axis sh

-- Dimension removal

-- | Remove a dimension after the caller has established that the axis exists.
type family RemoveAxisAt (axis :: Nat) (sh :: [Nat]) :: [Nat] where
    RemoveAxisAt 0 (dim ': rest) = rest
    RemoveAxisAt axis (dim ': rest) = dim ': RemoveAxisAt (axis - 1) rest

-- | Check the axis bound before removing its dimension.
type family RemoveAxisChecked (order :: Ordering) (axis :: Nat) (sh :: [Nat]) :: [Nat] where
    RemoveAxisChecked 'LT axis sh = RemoveAxisAt axis sh
    RemoveAxisChecked order axis sh =
        TypeError ('Text "Cannot remove axis " ':<>: 'ShowType axis ':<>: 'Text " from shape " ':<>: 'ShowType sh)

{- | Remove the dimension at an existing axis, regardless of its size.
@RemoveAxis 1 '[2, 3, 4]@ is @'[2, 4]@.
This family only computes a shape; it does not establish that squeezing the axis preserves elements.
-}
type RemoveAxis axis sh = RemoveAxisChecked (CmpNat axis (Rank sh)) axis sh

-- Dimension insertion

-- | Insert a dimension after the caller has checked the insertion position.
type family InsertAxisAt (axis :: Nat) (dim :: Nat) (sh :: [Nat]) :: [Nat] where
    InsertAxisAt 0 dim sh = dim ': sh
    InsertAxisAt axis dim (head ': rest) = head ': InsertAxisAt (axis - 1) dim rest

-- | Reject positions beyond the rank before inserting a dimension.
type family InsertAxisChecked (order :: Ordering) (axis :: Nat) (dim :: Nat) (sh :: [Nat]) :: [Nat] where
    InsertAxisChecked 'GT axis dim sh =
        TypeError ('Text "Cannot insert axis " ':<>: 'ShowType axis ':<>: 'Text " into shape " ':<>: 'ShowType sh)
    InsertAxisChecked order axis dim sh = InsertAxisAt axis dim sh

{- | Insert a dimension at a position from zero through the rank, inclusive.
@InsertAxis 1 5 '[2, 3]@ is @'[2, 5, 3]@.
Insertion into a scalar is permitted only at position zero.
-}
type InsertAxis axis dim sh = InsertAxisChecked (CmpNat axis (Rank sh)) axis dim sh

-- Axis selection and permutation

-- | Require an existing axis by comparing its number with the input rank.
type family CheckAxis (order :: Ordering) (axis :: Nat) (sh :: [Nat]) :: Constraint where
    CheckAxis 'LT axis sh = ()
    CheckAxis order axis sh =
        TypeError ('Text "Axis " ':<>: 'ShowType axis ':<>: 'Text " is out of bounds for shape " ':<>: 'ShowType sh)

-- | Reject an axis that also occurs later in the selection.
type family CheckDuplicate (duplicate :: Bool) (axis :: Nat) :: Constraint where
    CheckDuplicate 'False axis = ()
    CheckDuplicate 'True axis =
        TypeError ('Text "Repeated axis " ':<>: 'ShowType axis ':<>: 'Text " in axis selection")

-- | Reject a repeated axis before selecting its dimension and processing the remaining axes.
type family SelectAxis (duplicate :: Bool) (axis :: Nat) (rest :: [Nat]) (sh :: [Nat]) :: [Nat] where
    SelectAxis 'True axis rest sh =
        TypeError ('Text "Repeated axis " ':<>: 'ShowType axis ':<>: 'Text " in axis selection")
    SelectAxis 'False axis rest sh = AxisSize axis sh ': SelectAxes rest sh

{- | Select dimensions in the requested order, with no repeated or out-of-bounds axes.
@SelectAxes '[2, 0] '[2, 3, 4]@ is @'[4, 2]@.
The selection may omit axes; an empty selection produces the scalar shape.
-}
type family SelectAxes (xs :: [Nat]) (sh :: [Nat]) :: [Nat] where
    SelectAxes '[] sh = '[]
    SelectAxes (axis ': rest) sh = SelectAxis (Contains axis rest) axis rest sh

{- | Require a possibly partial selection of unique, existing axes.
The empty selection is valid for any shape; the order of selected axes is unrestricted.
-}
type family CanSelectAxes (xs :: [Nat]) (sh :: [Nat]) :: Constraint where
    CanSelectAxes '[] sh = ()
    CanSelectAxes (axis ': rest) sh =
        (CheckAxis (CmpNat axis (Rank sh)) axis sh, CheckDuplicate (Contains axis rest) axis, CanSelectAxes rest sh)

-- | Require one selected axis per input dimension before validating the selection.
type family PermuteAxesChecked (order :: Ordering) (xs :: [Nat]) (sh :: [Nat]) :: [Nat] where
    PermuteAxesChecked 'EQ xs sh = SelectAxes xs sh
    PermuteAxesChecked order xs sh =
        TypeError
            ( 'Text "Axis permutation "
                ':<>: 'ShowType xs
                ':<>: 'Text " must contain every axis of shape "
                ':<>: 'ShowType sh
            )

{- | Reorder dimensions using every input axis exactly once.
@PermuteAxes '[2, 0, 1] '[2, 3, 4]@ is @'[4, 2, 3]@.
Missing, repeated, and out-of-bounds axes produce type errors.
-}
type PermuteAxes xs sh = PermuteAxesChecked (CmpNat (Rank xs) (Rank sh)) xs sh

-- Singleton dimensions

-- | Require the selected dimension to have size one.
type family CheckSingleton (axis :: Nat) (sh :: [Nat]) (dim :: Nat) :: Constraint where
    CheckSingleton axis sh 1 = ()
    CheckSingleton axis sh dim =
        TypeError
            ( 'Text "Cannot squeeze axis "
                ':<>: 'ShowType axis
                ':<>: 'Text " of shape "
                ':<>: 'ShowType sh
                ':<>: 'Text ": expected size 1, got "
                ':<>: 'ShowType dim
            )

{- | Require an existing axis of size one, so that removing it preserves the number of elements.
Zero-sized dimensions cannot be squeezed.
-}
type CanSqueeze axis sh = CheckSingleton axis sh (AxisSize axis sh)

-- Matrix transposition

-- | Swap the two leading dimensions of a reversed shape and restore left-to-right order.
type family TransposeReversed (sh :: [Nat]) :: [Nat] where
    TransposeReversed (cols ': rows ': batch) = Reverse (rows ': cols ': batch)
    TransposeReversed sh =
        TypeError ('Text "Transpose requires tensor rank at least two, got " ':<>: 'ShowType (Reverse sh))

{- | Swap the last two dimensions while preserving all leading batch dimensions.
@TransposeShape '[5, 2, 3]@ is @'[5, 3, 2]@.
Inputs must have rank at least two.
-}
type TransposeShape sh = TransposeReversed (Reverse sh)

-- Concatenation

-- | Add the selected dimensions and require all other dimensions to match.
type family ConcatShapeAt (axis :: Nat) (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
    ConcatShapeAt 0 (left ': rest) (right ': rest) = (left + right) ': rest
    ConcatShapeAt 0 lhs rhs =
        TypeError ('Text "Concat dimensions outside the selected axis must match")
    ConcatShapeAt axis (dim ': left) (dim ': right) = dim ': ConcatShapeAt (axis - 1) left right
    ConcatShapeAt axis lhs rhs =
        TypeError ('Text "Concat dimensions outside the selected axis must match")

-- | Check that the concatenation axis exists before combining dimensions.
type family ConcatAxisChecked (order :: Ordering) (axis :: Nat) (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
    ConcatAxisChecked 'LT axis lhs rhs = ConcatShapeAt axis lhs rhs
    ConcatAxisChecked order axis lhs rhs =
        TypeError
            ('Text "Concat axis " ':<>: 'ShowType axis ':<>: 'Text " is out of bounds for shape " ':<>: 'ShowType lhs)

-- | Check that concatenated shapes have the same rank before checking their dimensions.
type family ConcatRankChecked (order :: Ordering) (axis :: Nat) (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
    ConcatRankChecked 'EQ axis lhs rhs = ConcatAxisChecked (CmpNat axis (Rank lhs)) axis lhs rhs
    ConcatRankChecked order axis lhs rhs =
        TypeError ('Text "Concat ranks must match: " ':<>: 'ShowType lhs ':<>: 'Text " and " ':<>: 'ShowType rhs)

{- | Shape obtained by concatenating equally ranked tensors along an existing axis.

Dimensions outside that axis must match; the selected dimensions are added.
@ConcatShape 1 '[2, 3] '[2, 4]@ is @'[2, 7]@.
Scalar concatenation and mismatched ranks or unselected dimensions produce type errors.
-}
type ConcatShape axis lhs rhs = ConcatRankChecked (CmpNat (Rank lhs) (Rank rhs)) axis lhs rhs

-- Reduction

-- | Omit a selected dimension or prepend an unselected dimension to the result.
type family KeepAxis (selected :: Bool) (dim :: Nat) (rest :: [Nat]) :: [Nat] where
    KeepAxis 'True dim rest = rest
    KeepAxis 'False dim rest = dim ': rest

-- | Reject an empty selected dimension for a reduction without an identity.
type family CheckReductionDimension (axis :: Nat) (sh :: [Nat]) (dim :: Nat) :: Constraint where
    CheckReductionDimension axis sh 0 =
        TypeError
            ( 'Text "Cannot reduce empty axis "
                ':<>: 'ShowType axis
                ':<>: 'Text " of shape "
                ':<>: 'ShowType sh
                ':<>: 'Text " without an identity"
            )
    CheckReductionDimension axis sh dim = ()

-- | Check that each selected dimension is nonempty, using the original input shape.
type family CheckReductionAxes (xs :: [Nat]) (sh :: [Nat]) :: Constraint where
    CheckReductionAxes '[] sh = ()
    CheckReductionAxes (axis ': rest) sh =
        (CheckReductionDimension axis sh (AxisSize axis sh), CheckReductionAxes rest sh)

-- | Reject a shape whose total element count is zero.
type family CheckNonEmpty (sh :: [Nat]) (size :: Nat) :: Constraint where
    CheckNonEmpty sh 0 =
        TypeError ('Text "Cannot reduce empty tensor of shape " ':<>: 'ShowType sh ':<>: 'Text " without an identity")
    CheckNonEmpty sh size = ()

-- | Visit the original axes in order, omitting dimensions named in the reduction list.
type family ReduceShapeAt (axis :: Nat) (xs :: [Nat]) (sh :: [Nat]) :: [Nat] where
    ReduceShapeAt axis xs '[] = '[]
    ReduceShapeAt axis xs (dim ': rest) =
        KeepAxis (Contains axis xs) dim (ReduceShapeAt (axis + 1) xs rest)

{- | Remove selected axes while retaining the original order of all remaining axes.

@ReduceShape '[0, 2] '[2, 3, 4]@ is @'[3]@.
An empty selection leaves the shape unchanged.
This family does not validate the selection; combine it with 'CanReduce' or 'CanReduceNonEmpty'.
-}
type ReduceShape xs sh = ReduceShapeAt 0 xs sh

{- | Require unique reduction axes within the original input shape.
Selected dimensions may have size zero; operations without an identity use 'CanReduceNonEmpty' instead.
-}
type CanReduce xs sh = CanSelectAxes xs sh

{- | Require valid reduction axes and at least one element along every selected axis.
Unselected dimensions may be empty, in which case the result is empty.
An empty selection satisfies this constraint even for an empty tensor.
-}
type CanReduceNonEmpty xs sh = (CanReduce xs sh, CheckReductionAxes xs sh)

{- | Require at least one tensor element for a full reduction without an identity.
Scalars satisfy this constraint because their element count is one.
Any shape containing a zero dimension is rejected.
-}
type NonEmpty sh = CheckNonEmpty sh (ShapeSize sh)
