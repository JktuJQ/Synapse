{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Tensor construction and initialisation.
module Synapse.Tensor.Construction
  ( arange,
    eye,
    fill,
    fromGen,
    full,
    fullLike,
    identity,
    linspace,
    logspace,
    ones,
    onesLike,
    scalar,
    zeros,
    zerosLike,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Elementwise (map)
import Synapse.Tensor.Ops.Index (coordinate)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shapeSize)
import Synapse.Tensor.Type (FromIntegral, Tensor (..), TensorElement (..), TensorIndex (..), constant)
import qualified Synapse.Tensor.Type as E
import Prelude hiding (map)

-- | Turn an element expression into a rank-zero tensor for execution.
-- Numeric literals work directly; use 'constant' for existing host values.
scalar :: (DType a) => TensorElement a -> Tensor '[] a
scalar = Tensor . A.unit . unTensorElement

-- | Generate each element from its complete multidimensional index.
-- The callback builds element expressions.
-- Use @coordinate \@axis@ to read an axis coordinate.
-- Scalar indices have no coordinates; empty tensors have no elements.
--
-- @
-- fromGen \@'[2, 3] (\ix -> coordinate \@0 ix * 10 + coordinate \@1 ix)
-- @
fromGen :: forall sh a. (Shape sh, DType a) => (TensorIndex sh -> TensorElement a) -> Tensor sh a
fromGen f = Tensor $ A.generate dims $ \ix ->
  unTensorElement (f (TensorIndex ix))
  where
    dims = A.constant (backendShape (Proxy @sh))

-- | Create a tensor filled with the same value in every element.
full :: forall sh a. (Shape sh, DType a) => a -> Tensor sh a
full x = Tensor $ A.fill (A.constant (backendShape (Proxy @sh))) (A.constant x)

-- | Alias for 'full'.
fill :: forall sh a. (Shape sh, DType a) => a -> Tensor sh a
fill = full @sh

-- | Create a tensor filled with zeros.
zeros :: forall sh a. (Shape sh, DType a, Num a) => Tensor sh a
zeros = full @sh 0

-- | Create a tensor filled with ones.
ones :: forall sh a. (Shape sh, DType a, Num a) => Tensor sh a
ones = full @sh 1

-- | Create a tensor with the same shape as another tensor, filled with a value.
fullLike :: forall sh a b. (Shape sh, DType a) => Tensor sh b -> a -> Tensor sh a
fullLike _ = full @sh

-- | Create a zero tensor with the same shape as another tensor.
zerosLike :: forall sh a b. (Shape sh, DType a, Num a) => Tensor sh b -> Tensor sh a
zerosLike _ = zeros @sh

-- | Create a ones tensor with the same shape as another tensor.
onesLike :: forall sh a b. (Shape sh, DType a, Num a) => Tensor sh b -> Tensor sh a
onesLike _ = ones @sh

-- | Create a vector with evenly spaced values @start, start + step, ...@.
arange :: forall n a. (Shape '[n], DType a, Num (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) => a -> a -> Tensor '[n] a
arange start step =
  fromGen @'[n] $ \ix -> constant start + E.fromIntegral (coordinate @0 ix) * constant step

-- | Create a vector with @n@ evenly spaced values from @start@ to @end@.
-- For @n >= 2@ both endpoints are included, with the last value taken directly from @end@ to avoid accumulated rounding.
-- Zero elements produce an empty vector; one element produces @start@.
-- Descending endpoints are supported.
linspace :: forall n a. (Shape '[n], DType a, Fractional a, Fractional (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) => a -> a -> Tensor '[n] a
linspace start end =
  case shapeSize @'[n] of
    0 -> full @'[n] start
    1 -> full @'[n] start
    count ->
      let step :: TensorElement a
          step = (constant end - constant start) / constant (fromIntegral (count - 1))
       in fromGen @'[n] $ \ix ->
            let i = unTensorElement (coordinate @0 ix)
                value = constant start + E.fromIntegral (coordinate @0 ix) * step
             in TensorElement (A.cond (i A.== A.constant (count - 1)) (A.constant end) (unTensorElement value))

-- | Exponentiate 'linspace' using the supplied base.
-- Parameters are start exponent, end exponent, then base.
-- Empty and singleton cases follow 'linspace'.
logspace :: forall n a. (Shape '[n], DType a, Floating a, Floating (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) => a -> a -> a -> Tensor '[n] a
logspace start end base =
  map (constant base **) (linspace @n @a start end)

-- | Create a matrix with ones on the main diagonal and zeros elsewhere.
eye ::
  forall rows cols a.
  (Shape '[rows, cols], DType a, Num (TensorElement a)) =>
  Tensor '[rows, cols] a
eye =
  fromGen @'[rows, cols] $ \ix ->
    let row = unTensorElement (coordinate @0 ix)
        col = unTensorElement (coordinate @1 ix)
     in TensorElement (A.cond (row A.== col) (unTensorElement (1 :: TensorElement a)) (unTensorElement (0 :: TensorElement a)))

-- | Create a square identity matrix.
identity :: forall n a. (Shape '[n, n], DType a, Num (TensorElement a)) => Tensor '[n, n] a
identity = eye @n @n
