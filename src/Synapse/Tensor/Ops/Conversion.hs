{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Conversion between numeric element types.
module Synapse.Tensor.Ops.Conversion
    ( FromIntegral (..)
    , ToFloating (..)
    ) where

import Synapse.Tensor.Ops.Raw.Elementwise (map)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Prelude hiding (fromIntegral, map)

import qualified Data.Array.Accelerate as A

-- | Conversion of integral expressions to numeric expressions.
class FromIntegral a b where
    -- | Convert an integral expression without evaluating it on the host.
    fromIntegral
        :: a
        -> b

instance (A.Integral a, A.FromIntegral a b) => FromIntegral (TensorElement a) (TensorElement b) where
    fromIntegral (TensorElement x) = TensorElement (A.fromIntegral x)
instance
    (out ~ sh, Shape sh, DType a, DType b, A.Integral a, A.FromIntegral a b)
    => FromIntegral (Tensor sh a) (Tensor out b)
    where
    fromIntegral (Tensor x) = Tensor (A.map A.fromIntegral x)

-- | Convert numeric element expressions or equally shaped tensors to a floating-point type.
class ToFloating a b where
    -- | Convert each numeric value to the target floating-point representation.
    toFloating
        :: a
        -> b

instance (A.Num a, A.Floating b, A.ToFloating a b) => ToFloating (TensorElement a) (TensorElement b) where
    toFloating (TensorElement x) = TensorElement (A.toFloating x)
instance
    (out ~ sh, Shape sh, DType a, DType b, ToFloating (TensorElement a) (TensorElement b))
    => ToFloating (Tensor sh a) (Tensor out b)
    where
    toFloating = map toFloating
