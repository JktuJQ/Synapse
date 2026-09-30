{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Tensor construction and host value conversion.
module Synapse.Tensor.Construction
  ( arange,
    eye,
    fill,
    fromGen,
    fromList,
    full,
    fullLike,
    identity,
    linspace,
    logspace,
    ones,
    onesLike,
    scalar,
    unsafeFromList,
    zeros,
    zerosLike,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shape (KnownShape, shape, shapeSize, shapeVal)
import Synapse.Tensor.Type (Tensor (..))

-- | Create a scalar tensor.
scalar :: (DType a) => a -> Tensor '[] a
scalar x = fromList @'[] [x]

-- | Create a tensor by generating each row-major element from its linear index.
fromGen :: forall sh a. (KnownShape sh, DType a) => (Int -> a) -> Tensor sh a
fromGen f = unsafeFromList @sh (map f [0 .. shapeSize @sh - 1])

-- | Create a tensor filled with the same value in every element.
full :: forall sh a. (KnownShape sh, DType a) => a -> Tensor sh a
full x = unsafeFromList @sh (replicate (shapeSize @sh) x)

-- | Alias for 'full'.
fill :: forall sh a. (KnownShape sh, DType a) => a -> Tensor sh a
fill = full @sh

-- | Create a tensor filled with zeros.
zeros :: forall sh a. (KnownShape sh, DType a, Num a) => Tensor sh a
zeros = full @sh 0

-- | Create a tensor filled with ones.
ones :: forall sh a. (KnownShape sh, DType a, Num a) => Tensor sh a
ones = full @sh 1

-- | Create a tensor with the same shape as another tensor, filled with a value.
fullLike :: forall sh a b. (KnownShape sh, DType a) => Tensor sh b -> a -> Tensor sh a
fullLike _ = full @sh

-- | Create a zero tensor with the same shape as another tensor.
zerosLike :: forall sh a b. (KnownShape sh, DType a, Num a) => Tensor sh b -> Tensor sh a
zerosLike _ = zeros @sh

-- | Create a ones tensor with the same shape as another tensor.
onesLike :: forall sh a b. (KnownShape sh, DType a, Num a) => Tensor sh b -> Tensor sh a
onesLike _ = ones @sh

-- | Create a vector with evenly spaced values @start, start + step, ...@.
arange :: forall n a. (KnownShape '[n], DType a, Num a) => a -> a -> Tensor '[n] a
arange start step =
  fromGen @'[n] $ \ix -> start + fromIntegral ix * step

-- | Create a vector with @n@ evenly spaced values from @start@ to @end@.
linspace :: forall n a. (KnownShape '[n], DType a, Fractional a) => a -> a -> Tensor '[n] a
linspace start end =
  case shapeSize @'[n] of
    0 -> unsafeFromList @'[n] []
    1 -> unsafeFromList @'[n] [start]
    count ->
      let step = (end - start) / fromIntegral (count - 1)
       in fromGen @'[n] $ \ix -> start + fromIntegral ix * step

-- | Create a vector with values spaced evenly on a log scale.
logspace :: forall n a. (KnownShape '[n], DType a, Floating a) => a -> a -> a -> Tensor '[n] a
logspace start end base =
  case shapeSize @'[n] of
    0 -> unsafeFromList @'[n] []
    1 -> unsafeFromList @'[n] [base ** start]
    count ->
      let step = (end - start) / fromIntegral (count - 1)
       in fromGen @'[n] $ \ix -> base ** (start + fromIntegral ix * step)

-- | Create a matrix with ones on the main diagonal and zeros elsewhere.
eye ::
  forall rows cols a.
  (KnownShape '[rows, cols], KnownNat cols, DType a, Num a) =>
  Tensor '[rows, cols] a
eye =
  fromGen @'[rows, cols] $ \ix ->
    let cols = fromInteger (natVal (Proxy @cols))
        (row, col) = ix `quotRem` cols
     in if row == col then 1 else 0

-- | Create a square identity matrix.
identity :: forall n a. (KnownShape '[n, n], KnownNat n, DType a, Num a) => Tensor '[n, n] a
identity = eye @n @n

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
