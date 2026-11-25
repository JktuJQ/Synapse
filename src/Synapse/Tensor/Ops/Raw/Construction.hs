{-# LANGUAGE TypeFamilies #-}

-- | Tensor construction from values and coordinate-local element expressions.
module Synapse.Tensor.Ops.Raw.Construction
    ( -- * Tensor construction
      fromGen
    , full
    , eye
    , arange
    , linspace
    , logspace

      -- * Element construction
    , element
    ) where

import Data.Array.Accelerate qualified as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Ops.Conversion (FromIntegral)
import Synapse.Tensor.Ops.Conversion qualified as E (fromIntegral)
import Synapse.Tensor.Ops.Raw.Elementwise (map)
import Synapse.Tensor.Ops.Raw.Index (coordinate)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..), TensorIndex (..))
import Synapse.Tensor.Shapes.Backend (backendShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Prelude hiding (map)
import Prelude qualified as P

-- | Generate each element from its multidimensional index.
fromGen
    :: forall sh a
     . (Shape sh, DType a)
    => (TensorIndex sh -> TensorElement a)
    -> Tensor sh a
fromGen f = Tensor $ A.generate dims $ \ix ->
    unTensorElement (f (TensorIndex ix))
  where
    dims = A.constant (backendShape (Proxy @sh))

-- | Create a tensor filled with the supplied value.
full
    :: forall sh a
     . (Shape sh, DType a)
    => a
    -> Tensor sh a
full x = Tensor $ A.fill (A.constant (backendShape (Proxy @sh))) (A.constant x)

-- | Compare row and column coordinates to construct a diagonal matrix.
eye
    :: forall rows cols a
     . (Shape '[rows, cols], DType a, Num (TensorElement a), P.Num a)
    => Tensor '[rows, cols] a
eye =
    fromGen @'[rows, cols] $ \ix ->
        let row = unTensorElement (coordinate @0 ix)
            col = unTensorElement (coordinate @1 ix)
         in TensorElement
                (A.cond (row A.== col) (unTensorElement (1 :: TensorElement a)) (unTensorElement (0 :: TensorElement a)))

-- | Construct an arithmetic progression from coordinate expressions.
arange
    :: forall n a
     . (Shape '[n], DType a, Num (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a), P.Num a)
    => a
    -> a
    -> Tensor '[n] a
arange start step =
    fromGen @'[n] $ \ix -> element start + E.fromIntegral (coordinate @0 ix :: TensorElement Int) * element step

-- | Interpolate endpoints, retaining them exactly and avoiding subtraction overflow.
linspace
    :: forall n a
     . ( Shape '[n]
       , DType a
       , Fractional a
       , Fractional (TensorElement a)
       , FromIntegral (TensorElement Int) (TensorElement a)
       )
    => a
    -> a
    -> Tensor '[n] a
linspace start end =
    case shapeSize @'[n] of
        0 -> full @'[n] start
        1 -> full @'[n] start
        count ->
            fromGen @'[n] $ \ix ->
                let i = unTensorElement (coordinate @0 ix)
                    weight = E.fromIntegral (coordinate @0 ix) / element (fromIntegral (count - 1))
                    value = (1 - weight) * element start + weight * element end
                 in TensorElement $
                        A.cond (i A.== 0) (A.constant start) $
                            A.cond (i A.== A.constant (count - 1)) (A.constant end) (unTensorElement value)

-- | Exponentiate evenly spaced exponents.
logspace
    :: forall n a
     . ( Shape '[n]
       , DType a
       , Floating a
       , Floating (TensorElement a)
       , FromIntegral (TensorElement Int) (TensorElement a)
       )
    => a
    -> a
    -> a
    -> Tensor '[n] a
logspace start end base = map (element base **) (linspace @n @a start end)

-- | Embed a value in an element expression.
element
    :: forall a
     . DType a
    => a
    -> TensorElement a
element = TensorElement . A.constant
