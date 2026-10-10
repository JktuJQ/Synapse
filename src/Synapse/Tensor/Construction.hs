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
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shapeSize)
import Synapse.Tensor.Type (FromIntegral, Tensor (..), TensorElement (..), constant)
import qualified Synapse.Tensor.Type as E

-- | Create a scalar tensor.
scalar :: (DType a) => TensorElement a -> Tensor '[] a
scalar = Tensor . A.unit . unTensorElement

-- | Generate elements from row-major linear indices.
fromGen :: forall sh a. (Shape sh, DType a) => (TensorElement Int -> TensorElement a) -> Tensor sh a
fromGen f = Tensor $ A.generate dims $ \ix ->
  unTensorElement (f (TensorElement (A.toIndex dims ix)))
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
  fromGen @'[n] $ \ix -> constant start + E.fromIntegral ix * constant step

-- | Create a vector with @n@ evenly spaced values from @start@ to @end@.
linspace :: forall n a. (Shape '[n], DType a, Fractional a, Fractional (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) => a -> a -> Tensor '[n] a
linspace start end =
  case shapeSize @'[n] of
    0 -> full @'[n] start
    1 -> full @'[n] start
    count ->
      let step :: TensorElement a
          step = (constant end - constant start) / constant (fromIntegral (count - 1))
       in fromGen @'[n] $ \ix -> constant start + E.fromIntegral ix * step

-- | Create a vector with values spaced evenly on a log scale.
logspace :: forall n a. (KnownNat n, Shape '[n], DType a, Floating a, Floating (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a)) => a -> a -> a -> Tensor '[n] a
logspace start end base =
  case linspace @n @a start end of
    Tensor exponents -> Tensor $ A.map (\x -> unTensorElement (constant base ** TensorElement x)) exponents

-- | Create a matrix with ones on the main diagonal and zeros elsewhere.
eye ::
  forall rows cols a.
  (Shape '[rows, cols], KnownNat cols, DType a, Num (TensorElement a)) =>
  Tensor '[rows, cols] a
eye =
  fromGen @'[rows, cols] $ \(TensorElement ix) ->
    let cols = A.constant (fromInteger (natVal (Proxy @cols)) :: Int)
        row = ix `quot` cols
        col = ix `rem` cols
     in TensorElement (A.cond (row A.== col) (unTensorElement (1 :: TensorElement a)) (unTensorElement (0 :: TensorElement a)))

-- | Create a square identity matrix.
identity :: forall n a. (Shape '[n, n], KnownNat n, DType a, Num (TensorElement a)) => Tensor '[n, n] a
identity = eye @n @n
