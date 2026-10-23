{-# LANGUAGE ScopedTypeVariables #-}

-- | Internal element transformations used to implement tensor primitives.
module Synapse.Tensor.Ops.Elementwise.Helpers (map, zipWith, zipWith3) where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))
import Prelude hiding (map, zipWith, zipWith3)

-- | Transform each element while retaining the tensor shape.
map :: (Shape sh, DType a, DType b) => (TensorElement a -> TensorElement b) -> Tensor sh a -> Tensor sh b
map f (Tensor x) = Tensor (A.map (unTensorElement . f . TensorElement) x)

-- | Combine corresponding elements of equally shaped tensors.
zipWith :: (Shape sh, DType a, DType b, DType c) => (TensorElement a -> TensorElement b -> TensorElement c) -> Tensor sh a -> Tensor sh b -> Tensor sh c
zipWith f (Tensor x) (Tensor y) = Tensor (A.zipWith (\a b -> unTensorElement (f (TensorElement a) (TensorElement b))) x y)

-- | Combine corresponding elements of three equally shaped tensors.
zipWith3 :: (Shape sh, DType a, DType b, DType c, DType d) => (TensorElement a -> TensorElement b -> TensorElement c -> TensorElement d) -> Tensor sh a -> Tensor sh b -> Tensor sh c -> Tensor sh d
zipWith3 f (Tensor x) (Tensor y) (Tensor z) = Tensor (A.zipWith3 (\a b c -> unTensorElement (f (TensorElement a) (TensorElement b) (TensorElement c))) x y z)
