{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Core tensor type and basic value conversion.
module Synapse.Tensor.Core
  ( Tensor (..),
    fromList,
    scalar,
    unsafeFromList,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (Nat)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shape (KnownShape (..), ShapeToDIM, shape, shapeSize)

-- | A staged tensor expression.
newtype Tensor (sh :: [Nat]) a
  = Tensor {unTensor :: A.Acc (A.Array (ShapeToDIM sh) a)}

-- | Create a scalar tensor.
scalar :: (DType a) => a -> Tensor '[] a
scalar x = fromList @'[] [x]

-- | Create a tensor from a row-major list without validating its length.
--
-- The caller must ensure that the list contains exactly the number of elements
-- required by the target shape.
unsafeFromList :: forall sh a. (KnownShape sh, DType a) => [a] -> Tensor sh a
unsafeFromList = Tensor . A.use . A.fromList (shapeVal (Proxy @sh))

-- | Create a tensor from a row-major list.
fromList :: forall sh a. (KnownShape sh, DType a) => [a] -> Tensor sh a
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
