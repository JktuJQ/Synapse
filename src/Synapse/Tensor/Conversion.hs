{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Conversion between host data, materialized tensors, and tensor computations.
module Synapse.Tensor.Conversion
  ( fromListData,
    fromList,
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

-- | Check the list length before returning a result, retaining the calling function's name in errors.
checkListLength :: forall sh proxy a result. (Shape sh) => proxy sh -> String -> [a] -> result -> result
checkListLength _ name xs result = check expectedSize (0 :: Int) xs
  where
    expectedSize = shapeSize @sh
    mismatch actual =
      error $
        "Synapse.Tensor."
          <> name
          <> ": expected "
          <> show expectedSize
          <> " element(s) for shape "
          <> show (shape @sh)
          <> ", but got "
          <> actual
    check 0 _ [] = result
    check 0 seen (_ : _) = mismatch ("at least " <> show (seen + 1))
    check _ seen [] = mismatch (show seen)
    check remaining seen (_ : rest) = check (remaining - 1) (seen + 1) rest

-- | Convert materialized tensor data to a row-major list.
toList :: (Shape sh, DType a) => TensorData sh a -> [a]
toList = A.toList . unTensorData

-- | Use materialized data as a constant input to a tensor computation without a list conversion.
-- The data must satisfy the shape invariant established by its constructor.
fromData :: forall {t} sh a. (Shape sh, TensorOps t a) => TensorData sh a -> t sh a
fromData = literal . Tensor . A.use . unTensorData

-- | Create materialized tensor data from a finite row-major list without running a backend.
-- The list must contain exactly as many elements as the shape requires; a length mismatch raises an error.
-- Suitable for preparing examples inside a dataset's loading function.
fromListData :: forall sh a. (Shape sh, DType a) => [a] -> TensorData sh a
fromListData xs = checkListLength (Proxy @sh) "fromListData" xs (TensorData (A.fromList (backendShape (Proxy @sh)) xs))

-- | Create a tensor from a row-major list without validating its length.
-- The caller must ensure that the list contains exactly the number of elements required by the target shape.
unsafeFromList :: forall {t} sh a. (Shape sh, TensorOps t a) => [a] -> t sh a
unsafeFromList = fromData . TensorData . A.fromList (backendShape (Proxy @sh))

-- | Create a tensor constant from a finite row-major list.
-- The list must contain exactly as many elements as the shape requires; a length mismatch raises an error.
fromList :: forall {t} sh a. (Shape sh, TensorOps t a) => [a] -> t sh a
fromList xs = checkListLength (Proxy @sh) "fromList" xs (unsafeFromList @sh xs)
