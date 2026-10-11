{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Shape-preserving elementwise tensor operations.
module Synapse.Tensor.Ops.Elementwise
  ( Boolean (..),
    Conditional (..),
    Eq (..),
    Ord (..),
    Integral (..),
    RealFloat (..),
    map,
    zip,
    zipWith,
    while,
    iterate,
    pair,
    fst,
    snd,
  )
where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))
import Prelude hiding (Eq, Integral, Ord, RealFloat, fst, isInfinite, isNaN, iterate, map, max, min, not, quot, rem, snd, zip, zipWith, (&&), (/=), (<), (<=), (==), (>), (>=), (||))

-- | Transform each element, preserving the shape while allowing a new element type.
map :: (Shape sh, DType a, DType b) => (TensorElement a -> TensorElement b) -> Tensor sh a -> Tensor sh b
map f (Tensor x) = Tensor (A.map (unTensorElement . f . TensorElement) x)

-- | Pair corresponding elements of two tensors with the same shape.
zip :: (Shape sh, DType a, DType b) => Tensor sh a -> Tensor sh b -> Tensor sh (a, b)
zip (Tensor x) (Tensor y) = Tensor (A.zip x y)

-- | Combine corresponding elements of two tensors with the same shape.
zipWith :: (Shape sh, DType a, DType b, DType c) => (TensorElement a -> TensorElement b -> TensorElement c) -> Tensor sh a -> Tensor sh b -> Tensor sh c
zipWith f (Tensor x) (Tensor y) = Tensor $ A.zipWith (\a b -> unTensorElement (f (TensorElement a) (TensorElement b))) x y

-- | Boolean operations on element expressions and equally shaped masks.
class Boolean a where
  -- | Negate each boolean value.
  not :: a -> a

  -- | Conjunction of corresponding boolean values.
  (&&) :: a -> a -> a

  -- | Disjunction of corresponding boolean values.
  (||) :: a -> a -> a

infixr 3 &&

infixr 2 ||

instance Boolean (TensorElement Bool) where
  not (TensorElement x) = TensorElement (A.not x)
  TensorElement x && TensorElement y = TensorElement (x A.&& y)
  TensorElement x || TensorElement y = TensorElement (x A.|| y)

instance (Shape sh) => Boolean (Tensor sh Bool) where
  not = map not
  (&&) = zipWith (&&)
  (||) = zipWith (||)

-- | Conditional selection of element expressions or tensor computations.
class Conditional a where
  -- | Boolean element or equally shaped tensor mask used by 'select'.
  type Condition a

  -- | Choose the first value where the condition is true and the second where it is false.
  -- Tensor masks select elementwise and all three tensor shapes must match.
  select :: Condition a -> a -> a -> a

  -- | Choose one branch using a scalar element condition.
  -- For tensors this branches on the whole computation rather than selecting individual elements.
  cond :: TensorElement Bool -> a -> a -> a

instance (DType a) => Conditional (TensorElement a) where
  type Condition (TensorElement a) = TensorElement Bool
  select = cond
  cond (TensorElement flag) (TensorElement x) (TensorElement y) = TensorElement (A.cond flag x y)

instance (Shape sh, DType a) => Conditional (Tensor sh a) where
  type Condition (Tensor sh a) = Tensor sh Bool
  select (Tensor mask) (Tensor x) (Tensor y) = Tensor (A.zipWith3 A.cond mask x y)
  cond (TensorElement flag) (Tensor x) (Tensor y) = Tensor (A.acond flag x y)

-- | Combine two element expressions into a pair, including loop states.
pair :: (DType a, DType b) => TensorElement a -> TensorElement b -> TensorElement (a, b)
pair (TensorElement x) (TensorElement y) = TensorElement (A.lift (x, y))

-- | Extract the first expression from an element pair.
fst :: (DType a, DType b) => TensorElement (a, b) -> TensorElement a
fst (TensorElement x) = TensorElement (A.fst x)

-- | Extract the second expression from an element pair.
snd :: (DType a, DType b) => TensorElement (a, b) -> TensorElement b
snd (TensorElement x) = TensorElement (A.snd x)

-- | Apply a state update while the predicate holds, testing it before every update.
-- The state type remains unchanged and the caller must ensure termination.
while :: (DType a) => (TensorElement a -> TensorElement Bool) -> (TensorElement a -> TensorElement a) -> TensorElement a -> TensorElement a
while predicate step (TensorElement initial) =
  TensorElement (A.while (unTensorElement . predicate . TensorElement) (unTensorElement . step . TensorElement) initial)

-- | Apply an element state update a specified number of times.
-- The iteration count is an element expression, so it can depend on the current tensor index.
iterate :: (DType a) => TensorElement Int -> (TensorElement a -> TensorElement a) -> TensorElement a -> TensorElement a
iterate (TensorElement count) step (TensorElement initial) =
  TensorElement (A.iterate count (unTensorElement . step . TensorElement) initial)

-- | Equality comparisons that produce element expressions or boolean tensors.
class Eq a where
  -- | Shape-preserving boolean result of a comparison.
  type Comparison a

  -- | Compare corresponding values for equality.
  (==) :: a -> a -> Comparison a

  -- | Compare corresponding values for inequality.
  (/=) :: a -> a -> Comparison a

infix 4 ==, /=

-- | Adapt a backend predicate to boolean element expressions.
elementComparison :: (A.Exp a -> A.Exp a -> A.Exp Bool) -> TensorElement a -> TensorElement a -> TensorElement Bool
elementComparison f (TensorElement x) (TensorElement y) = TensorElement (f x y)

instance (A.Eq a) => Eq (TensorElement a) where
  type Comparison (TensorElement a) = TensorElement Bool
  (==) = elementComparison (A.==)
  x /= y = not (x == y)

-- | Ordered comparisons and elementwise extrema without executing the computation.
-- Floating point comparisons follow IEEE semantics and extrema propagate NaN.
class (Eq a) => Ord a where
  -- | Compare corresponding values using strict less-than.
  (<) :: a -> a -> Comparison a

  -- | Compare corresponding values using less-than or equality.
  (<=) :: a -> a -> Comparison a

  -- | Compare corresponding values using strict greater-than.
  (>) :: a -> a -> Comparison a

  -- | Compare corresponding values using greater-than or equality.
  (>=) :: a -> a -> Comparison a

  -- | Select the smaller corresponding value.
  min :: a -> a -> a

  -- | Select the larger corresponding value.
  max :: a -> a -> a

infix 4 <, <=, >, >=

instance {-# OVERLAPPABLE #-} (A.Ord a) => Ord (TensorElement a) where
  (<) = elementComparison (A.<)
  (<=) = elementComparison (A.<=)
  (>) = elementComparison (A.>)
  (>=) = elementComparison (A.>=)
  min (TensorElement x) (TensorElement y) = TensorElement (A.min x y)
  max (TensorElement x) (TensorElement y) = TensorElement (A.max x y)

-- | Boolean ordering uses False before True.
instance {-# OVERLAPPING #-} Ord (TensorElement Bool) where
  x < y = not x && y
  x <= y = not x || y
  x > y = x && not y
  x >= y = x || not y
  min = (&&)
  max = (||)

-- | Preserve NaN before applying a floating point extremum operation.
nanExtremum :: (A.RealFloat a) => (A.Exp a -> A.Exp a -> A.Exp a) -> TensorElement a -> TensorElement a -> TensorElement a
nanExtremum f (TensorElement x) (TensorElement y) =
  TensorElement (A.cond (A.isNaN x A.|| A.isNaN y) (0 / 0) (f x y))

instance {-# OVERLAPPING #-} Ord (TensorElement Float) where
  (<) = elementComparison (A.<)
  (<=) = elementComparison (A.<=)
  (>) = elementComparison (A.>)
  (>=) = elementComparison (A.>=)
  min = nanExtremum A.min
  max = nanExtremum A.max

instance {-# OVERLAPPING #-} Ord (TensorElement Double) where
  (<) = elementComparison (A.<)
  (<=) = elementComparison (A.<=)
  (>) = elementComparison (A.>)
  (>=) = elementComparison (A.>=)
  min = nanExtremum A.min
  max = nanExtremum A.max

instance (Shape sh, DType a, Eq (TensorElement a)) => Eq (Tensor sh a) where
  type Comparison (Tensor sh a) = Tensor sh Bool
  (==) = zipWith (==)
  (/=) = zipWith (/=)

instance (Shape sh, DType a, Ord (TensorElement a)) => Ord (Tensor sh a) where
  (<) = zipWith (<)
  (<=) = zipWith (<=)
  (>) = zipWith (>)
  (>=) = zipWith (>=)
  min = zipWith min
  max = zipWith max

-- | Integer division with truncation toward zero and its corresponding remainder.
-- Division by zero is undefined and must be avoided by the caller.
class Integral a where
  -- | Divide corresponding values, rounding the quotient toward zero.
  quot :: a -> a -> a

  -- | Remainder whose sign follows the dividend.
  rem :: a -> a -> a

instance (A.Integral a) => Integral (TensorElement a) where
  quot (TensorElement x) (TensorElement y) = TensorElement (A.quot x y)
  rem (TensorElement x) (TensorElement y) = TensorElement (A.rem x y)

instance (Shape sh, DType a, Integral (TensorElement a)) => Integral (Tensor sh a) where
  quot = zipWith quot
  rem = zipWith rem

-- | Classification of floating point elements and tensors.
class (Eq a) => RealFloat a where
  -- | Test whether each value is NaN.
  isNaN :: a -> Comparison a

  -- | Test whether each value is positive or negative infinity.
  isInfinite :: a -> Comparison a

instance (A.RealFloat a) => RealFloat (TensorElement a) where
  isNaN (TensorElement x) = TensorElement (A.isNaN x)
  isInfinite (TensorElement x) = TensorElement (A.isInfinite x)

instance (Shape sh, DType a, RealFloat (TensorElement a)) => RealFloat (Tensor sh a) where
  isNaN = map isNaN
  isInfinite = map isInfinite
