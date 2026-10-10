{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Shape-preserving elementwise tensor operations.
module Synapse.Tensor.Ops.Elementwise (Eq (..), Ord (..), map, zip, zipWith) where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))
import Prelude hiding (Eq, Ord, map, max, min, zip, zipWith, (/=), (<), (<=), (==), (>), (>=))

-- | Transform each element, preserving the shape while allowing a new element type.
map :: (Shape sh, DType a, DType b) => (TensorElement a -> TensorElement b) -> Tensor sh a -> Tensor sh b
map f (Tensor x) = Tensor (A.map (unTensorElement . f . TensorElement) x)

-- | Pair corresponding elements of two tensors with the same shape.
zip :: (Shape sh, DType a, DType b) => Tensor sh a -> Tensor sh b -> Tensor sh (a, b)
zip (Tensor x) (Tensor y) = Tensor (A.zip x y)

-- | Combine corresponding elements of two tensors with the same shape.
zipWith :: (Shape sh, DType a, DType b, DType c) => (TensorElement a -> TensorElement b -> TensorElement c) -> Tensor sh a -> Tensor sh b -> Tensor sh c
zipWith f (Tensor x) (Tensor y) = Tensor $ A.zipWith (\a b -> unTensorElement (f (TensorElement a) (TensorElement b))) x y

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
  (/=) = elementComparison (\x y -> A.not (x A.== y))

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
  (<) = elementComparison (\x y -> A.not x A.&& y)
  (<=) = elementComparison (\x y -> A.not x A.|| y)
  (>) = elementComparison (\x y -> x A.&& A.not y)
  (>=) = elementComparison (\x y -> x A.|| A.not y)
  min (TensorElement x) (TensorElement y) = TensorElement (x A.&& y)
  max (TensorElement x) (TensorElement y) = TensorElement (x A.|| y)

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
