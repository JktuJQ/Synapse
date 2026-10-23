{-# LANGUAGE DataKinds #-}
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
  )
where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Elementwise.Helpers (map, zipWith, zipWith3)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))
import Prelude hiding (Eq, Integral, Ord, RealFloat, isInfinite, isNaN, map, max, min, not, quot, rem, zipWith, zipWith3, (&&), (/=), (<), (<=), (==), (>), (>=), (||))

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

  -- | Scalar boolean expression used to choose an entire computation.
  type Predicate a

  -- | Choose the first value where the condition is true and the second where it is false.
  -- Tensor masks select elementwise and all three tensor shapes must match.
  -- Differentiable values propagate gradients through the selected branches; the mask has no gradient history.
  select :: Condition a -> a -> a -> a

  -- | Choose one branch using a scalar condition.
  -- For tensors this branches on the whole computation rather than selecting individual elements.
  cond :: Predicate a -> a -> a -> a

instance (DType a) => Conditional (TensorElement a) where
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
  -- Differentiable floating point inputs share the gradient equally at ties.
  min :: a -> a -> a

  -- | Select the larger corresponding value.
  -- Differentiable floating point inputs share the gradient equally at ties.
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
