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
import Synapse.Tensor.Ops (TensorOps (literal))
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shape, shapeSize)
import Synapse.Tensor.Type (DType, Tensor (..), TensorData (..))

-- | Create a tensor from a row-major list without validating its length.
--
-- The caller must ensure that the list contains exactly the number of elements required by the target shape.
unsafeFromList :: forall {t} sh a. (Shape sh, TensorOps t a) => [a] -> t sh a
unsafeFromList = literal . Tensor . A.use . A.fromList (backendShape (Proxy @sh))

-- | Create a tensor constant from a row-major list.
-- The list must contain exactly as many elements as the shape requires; a length mismatch raises an error.
fromList :: forall {t} sh a. (Shape sh, TensorOps t a) => [a] -> t sh a
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

-- | Use materialized data as a constant input to a tensor computation without a list conversion.
-- The data must satisfy the shape invariant established by its constructor.
fromData :: forall {t} sh a. (Shape sh, TensorOps t a) => TensorData sh a -> t sh a
fromData = literal . Tensor . A.use . unTensorData
