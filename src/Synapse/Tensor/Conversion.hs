{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Conversion between host data, materialized tensors, and tensor computations.
module Synapse.Tensor.Conversion
  ( fromList,
    unsafeFromList,
    toList,
    fromData,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape, shape, shapeSize, shapeVal)
import Synapse.Tensor.Type (Tensor (..), TensorData (..))

-- | Create a tensor from a row-major list without validating its length.
--
-- The caller must ensure that the list contains exactly the number of elements
-- required by the target shape.
unsafeFromList :: forall sh a. (Shape sh, DType a) => [a] -> Tensor sh a
unsafeFromList = Tensor . A.use . A.fromList (shapeVal (Proxy @sh))

-- | Create a tensor from a row-major list.
fromList :: forall sh a. (Shape sh, DType a) => [a] -> Tensor sh a
fromList xs
  | actualSize == expectedSize =
      unsafeFromList @sh xs
  | otherwise =
      error $
        "Synapse.Tensor.fromList: expected "
          <> show expectedSize
          <> " element(s) for shape "
          <> show (shape @sh)
          <> ", but got "
          <> show actualSize
  where
    actualSize = length xs
    expectedSize = shapeSize @sh

-- | Convert materialized tensor data to a row-major list.
toList :: (Shape sh, DType a) => TensorData sh a -> [a]
toList = A.toList . unTensorData

-- | Use materialized data as input to a tensor computation without a list conversion.
-- The data must satisfy the shape invariant established by its constructor.
fromData :: (Shape sh, DType a) => TensorData sh a -> Tensor sh a
fromData = Tensor . A.use . unTensorData
