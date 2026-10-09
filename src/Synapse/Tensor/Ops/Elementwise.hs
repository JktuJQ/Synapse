{-# LANGUAGE FlexibleContexts #-}

-- | Shape-preserving elementwise tensor operations.
module Synapse.Tensor.Ops.Elementwise (map, zip, zipWith) where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))
import Prelude hiding (map, zip, zipWith)

-- | Transform each element, preserving the shape while allowing a new element type.
map :: (Shape sh, DType a, DType b) => (TensorElement a -> TensorElement b) -> Tensor sh a -> Tensor sh b
map f (Tensor x) = Tensor (A.map (unTensorElement . f . TensorElement) x)

-- | Pair corresponding elements of two tensors with the same shape.
zip :: (Shape sh, DType a, DType b) => Tensor sh a -> Tensor sh b -> Tensor sh (a, b)
zip (Tensor x) (Tensor y) = Tensor (A.zip x y)

-- | Combine corresponding elements of two tensors with the same shape.
zipWith :: (Shape sh, DType a, DType b, DType c) => (TensorElement a -> TensorElement b -> TensorElement c) -> Tensor sh a -> Tensor sh b -> Tensor sh c
zipWith f (Tensor x) (Tensor y) = Tensor $ A.zipWith (\a b -> unTensorElement (f (TensorElement a) (TensorElement b))) x y
