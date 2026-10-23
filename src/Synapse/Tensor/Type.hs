{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Core tensor type.
module Synapse.Tensor.Type
  ( ElementConstant (..),
    Tensor (..),
    Element,
    TensorElement (..),
    TensorIndex (..),
    TensorData (..),
    constant,
    FromIntegral (..),
  )
where

import qualified Data.Array.Accelerate as A
import Data.Kind (Type)
import GHC.TypeLits (Nat)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM)
import Prelude hiding (fromIntegral)

-- | Embedding host values into element expressions.
class ElementConstant element where
  -- | Embed a host value in an element expression.
  constantOp ::
    forall a.
    (DType a) =>
    a ->
    element a

instance ElementConstant TensorElement where
  constantOp = TensorElement . A.constant

-- | Embed a host value in an element expression.
constant :: forall a. (DType a) => a -> TensorElement a
constant = constantOp @TensorElement @a

-- | A staged tensor expression.
--
-- The constructor and 'unTensor' provide direct access to <https://hackage.haskell.org/package/accelerate Accelerate>.
-- When using this escape hatch, callers must ensure runtime dimensions match the type-level shape.
-- Subsequent Synapse operations rely on this invariant.
newtype Tensor (sh :: [Nat]) a
  = Tensor {unTensor :: A.Acc (A.Array (ShapeToDIM sh) a)}

-- | An element expression evaluated inside a tensor computation.
newtype TensorElement a
  = TensorElement {unTensorElement :: A.Exp a}

-- | Element expressions associated with a tensor representation.
-- Each element representation identifies its corresponding tensor representation.
type family Element (t :: [Nat] -> Type -> Type) = (e :: Type -> Type) | e -> t

type instance Element Tensor = TensorElement

-- | A complete multidimensional index for a tensor of shape @sh@.
-- Generators supply in-bounds indices.
-- Direct construction is an escape hatch: callers are responsible for the coordinate bounds of any custom index.
newtype TensorIndex (sh :: [Nat])
  = TensorIndex {unTensorIndex :: A.Exp (ShapeToDIM sh)}

deriving instance (Num (A.Exp a)) => Num (TensorElement a)

deriving instance (Fractional (A.Exp a)) => Fractional (TensorElement a)

deriving instance (Floating (A.Exp a)) => Floating (TensorElement a)

-- | Materialized input or output data.
-- Direct construction requires runtime dimensions to match the type-level shape, just as for 'Tensor'.
-- The underlying array can be converted using <https://hackage.haskell.org/package/accelerate Accelerate>.
newtype TensorData (sh :: [Nat]) a
  = TensorData {unTensorData :: A.Array (ShapeToDIM sh) a}

-- | Conversion of integral expressions to numeric expressions.
class FromIntegral a b where
  -- | Convert an integral expression without evaluating it on the host.
  fromIntegral :: a -> b

instance (A.Integral a, A.FromIntegral a b) => FromIntegral (TensorElement a) (TensorElement b) where
  fromIntegral (TensorElement x) = TensorElement (A.fromIntegral x)

instance (out ~ sh, Shape sh, DType a, DType b, A.Integral a, A.FromIntegral a b) => FromIntegral (Tensor sh a) (Tensor out b) where
  fromIntegral (Tensor x) = Tensor (A.map A.fromIntegral x)
