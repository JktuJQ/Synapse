{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Boolean masks, comparisons, and conditional selection.
module Synapse.Tensor.Ops.Logic
    ( -- * Masks and selection
      Boolean (..)
    , Conditional (..)

      -- * Comparison
    , Eq (..)
    , Ord (..)
    ) where

import Synapse.Tensor.Ops.Raw.Elementwise (map, zipWith, zipWith3)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Prelude hiding
    ( Eq
    , Ord
    , compare
    , map
    , max
    , min
    , not
    , zipWith
    , zipWith3
    , (&&)
    , (/=)
    , (<)
    , (<=)
    , (==)
    , (>)
    , (>=)
    , (||)
    )

import qualified Data.Array.Accelerate as A
import qualified Prelude as P

-- | Boolean operations on element expressions and equally shaped masks.
class Boolean a where
    -- | Negate each boolean value.
    not
        :: a
        -> a

    -- | Conjunction of corresponding boolean values.
    (&&)
        :: a
        -> a
        -> a

    -- | Disjunction of corresponding boolean values.
    (||)
        :: a
        -> a
        -> a

infixr 3 &&

infixr 2 ||

instance Boolean (TensorElement Bool) where
    not (TensorElement x) = TensorElement (A.not x)
    TensorElement x && TensorElement y = TensorElement (x A.&& y)
    TensorElement x || TensorElement y = TensorElement (x A.|| y)
instance Shape sh => Boolean (Tensor sh Bool) where
    not = map not
    (&&) = zipWith (&&)
    (||) = zipWith (||)

-- | Conditional selection of element expressions or tensor computations.
class Conditional a where
    -- | Boolean element or equally shaped tensor mask used by 'select'.
    type Condition a

    -- | Scalar boolean expression used to choose an entire computation.
    type Predicate a

    -- | Choose the first value where the condition is true and the second where it is false.
    select
        :: Condition a
        -> a
        -> a
        -> a

    -- | Choose one branch using a scalar condition.
    cond
        :: Predicate a
        -> a
        -> a
        -> a

instance DType a => Conditional (TensorElement a) where
    type Condition (TensorElement a) = TensorElement Bool
    type Predicate (TensorElement a) = TensorElement Bool
    select = cond
    cond (TensorElement flag) (TensorElement x) (TensorElement y) = TensorElement (A.cond flag x y)
instance (Shape sh, DType a) => Conditional (Tensor sh a) where
    type Condition (Tensor sh a) = Tensor sh Bool
    type Predicate (Tensor sh a) = Tensor '[] Bool
    select = zipWith3 select
    cond (Tensor flag) (Tensor x) (Tensor y) = Tensor (A.acond (A.the flag) x y)

-- | Equality comparisons that produce element expressions or boolean tensors.
class Eq a where
    -- | Shape-preserving boolean result of a comparison.
    type Comparison a

    -- | Compare corresponding values for equality.
    (==)
        :: a
        -> a
        -> Comparison a

    -- | Compare corresponding values for inequality.
    (/=)
        :: a
        -> a
        -> Comparison a

infix 4 ==, /=

-- | Convert a predicate on scalar expressions to boolean element expressions.
elementComparison
    :: (A.Exp a -> A.Exp a -> A.Exp Bool)
    -> TensorElement a
    -> TensorElement a
    -> TensorElement Bool
elementComparison f (TensorElement x) (TensorElement y) = TensorElement (f x y)

instance A.Eq a => Eq (TensorElement a) where
    type Comparison (TensorElement a) = TensorElement Bool
    (==) = elementComparison (A.==)
    x /= y = not (x == y)

{- | Ordered comparisons and elementwise extrema without executing the computation.
Floating point comparisons follow IEEE semantics and extrema propagate NaN.
-}
class Eq a => Ord a where
    -- | Element expressions or tensors containing Ordering values with the input shape.
    type OrderingResult a

    -- | Compare corresponding values, returning LT, EQ, or GT.
    compare
        :: a
        -> a
        -> OrderingResult a

    -- | Compare corresponding values using strict less-than.
    (<)
        :: a
        -> a
        -> Comparison a

    -- | Compare corresponding values using less-than or equality.
    (<=)
        :: a
        -> a
        -> Comparison a

    -- | Compare corresponding values using strict greater-than.
    (>)
        :: a
        -> a
        -> Comparison a

    -- | Compare corresponding values using greater-than or equality.
    (>=)
        :: a
        -> a
        -> Comparison a

    -- | Select the smaller corresponding value.
    min
        :: a
        -> a
        -> a

    -- | Select the larger corresponding value.
    max
        :: a
        -> a
        -> a

infix 4 <, <=, >, >=

instance {-# OVERLAPPABLE #-} A.Ord a => Ord (TensorElement a) where
    type OrderingResult (TensorElement a) = TensorElement Ordering
    compare (TensorElement x) (TensorElement y) = TensorElement (A.compare x y)
    (<) = elementComparison (A.<)
    (<=) = elementComparison (A.<=)
    (>) = elementComparison (A.>)
    (>=) = elementComparison (A.>=)
    min (TensorElement x) (TensorElement y) = TensorElement (A.min x y)
    max (TensorElement x) (TensorElement y) = TensorElement (A.max x y)

-- | Boolean ordering uses False before True.
instance {-# OVERLAPPING #-} Ord (TensorElement Bool) where
    type OrderingResult (TensorElement Bool) = TensorElement Ordering
    compare x y =
        select (x == y) (TensorElement $ A.constant EQ) $
            select x (TensorElement $ A.constant GT) (TensorElement $ A.constant LT)
    x < y = not x && y
    x <= y = not x || y
    x > y = x && not y
    x >= y = x || not y
    min = (&&)
    max = (||)

-- | Preserve NaN before applying a floating point extremum operation.
nanExtremum
    :: A.RealFloat a
    => (A.Exp a -> A.Exp a -> A.Exp a)
    -> TensorElement a
    -> TensorElement a
    -> TensorElement a
nanExtremum f (TensorElement x) (TensorElement y) =
    TensorElement (A.cond (A.isNaN x A.|| A.isNaN y) (0 / 0) (f x y))

instance {-# OVERLAPPING #-} Ord (TensorElement Float) where
    type OrderingResult (TensorElement Float) = TensorElement Ordering
    compare (TensorElement x) (TensorElement y) = TensorElement (A.compare x y)
    (<) = elementComparison (A.<)
    (<=) = elementComparison (A.<=)
    (>) = elementComparison (A.>)
    (>=) = elementComparison (A.>=)
    min = nanExtremum A.min
    max = nanExtremum A.max
instance {-# OVERLAPPING #-} Ord (TensorElement Double) where
    type OrderingResult (TensorElement Double) = TensorElement Ordering
    compare (TensorElement x) (TensorElement y) = TensorElement (A.compare x y)
    (<) = elementComparison (A.<)
    (<=) = elementComparison (A.<=)
    (>) = elementComparison (A.>)
    (>=) = elementComparison (A.>=)
    min = nanExtremum A.min
    max = nanExtremum A.max
instance (Shape sh, DType a, Eq (TensorElement a), P.Eq a) => Eq (Tensor sh a) where
    type Comparison (Tensor sh a) = Tensor sh Bool
    (==) = zipWith (==)
    (/=) = zipWith (/=)
instance
    (Shape sh, DType a, Ord (TensorElement a), OrderingResult (TensorElement a) ~ TensorElement Ordering, P.Ord a)
    => Ord (Tensor sh a)
    where
    type OrderingResult (Tensor sh a) = Tensor sh Ordering
    compare = zipWith compare
    (<) = zipWith (<)
    (<=) = zipWith (<=)
    (>) = zipWith (>)
    (>=) = zipWith (>=)
    min = zipWith min
    max = zipWith max
