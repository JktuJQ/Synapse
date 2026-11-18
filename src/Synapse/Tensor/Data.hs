{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Conversion between tensors, tensor data, and row-major lists.
module Synapse.Tensor.Data
    ( -- * Materialized data
      fromListData
    , toList

      -- * Staged constants
    , fromData
    , fromList

      -- * Unchecked construction
    , unsafeFromList
    )
where

import Data.Proxy (Proxy (..))

import Synapse.Tensor.Ops.Definition (TensorOps (fromTensor))
import Synapse.Tensor.Representation (DType, Tensor (..), TensorData (..), TensorLike)
import Synapse.Tensor.Shapes.Backend (backendShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shape, shapeSize)

import qualified Data.Array.Accelerate as A

-- | Validate the list length before returning the supplied result.
checkListLength
    :: forall sh proxy a result
     . Shape sh
    => proxy sh
    -> String
    -> [a]
    -> result
    -> result
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

-- | Construct tensor data from row-major elements. The list length must match the shape.
fromListData
    :: forall sh a
     . (Shape sh, DType a)
    => [a]
    -> TensorData sh a
fromListData xs =
    checkListLength (Proxy @sh) "fromListData" xs $
        TensorData $
            A.fromList (backendShape (Proxy @sh)) xs

-- | Return tensor data as a row-major list.
toList
    :: (Shape sh, DType a)
    => TensorData sh a
    -> [a]
toList = A.toList . unTensorData

-- | Convert tensor data to a constant tensor.
fromData
    :: forall {t :: TensorLike} sh a
     . (Shape sh, TensorOps t a)
    => TensorData sh a
    -> t sh a
fromData = fromTensor . Tensor . A.use . unTensorData

-- | Construct a tensor from row-major elements. The list length must match the shape.
fromList
    :: forall {t :: TensorLike} sh a
     . (Shape sh, TensorOps t a)
    => [a]
    -> t sh a
fromList xs = checkListLength (Proxy @sh) "fromList" xs (unsafeFromList @sh xs)

-- | Construct a tensor from row-major elements without checking the list length.
unsafeFromList
    :: forall {t :: TensorLike} sh a
     . (Shape sh, TensorOps t a)
    => [a]
    -> t sh a
unsafeFromList = fromData . TensorData . A.fromList (backendShape (Proxy @sh))
