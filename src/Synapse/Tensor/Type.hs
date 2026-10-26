{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Staged tensors, materialized data, element expressions, and representation correspondences.
module Synapse.Tensor.Type
  ( DType,
    Float32,
    Float64,
    WithTensor,
    Tensor (..),
    TensorLike,
    Element,
    TensorElement (..),
    TensorIndex (..),
    TensorData (..),
    element,
    FromIntegral (..),
  )
where

import qualified Data.Array.Accelerate as A
import Data.Int (Int32, Int64)
import Data.Kind (Type)
import Data.Word (Word32, Word64)
import GHC.TypeLits (Nat)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM)
import Prelude hiding (fromIntegral)

-- | Single-precision floating point tensor element.
type Float32 = Float

-- | Double-precision floating point tensor element.
type Float64 = Double

-- | Element types supported by Synapse tensors.
class (A.Elt a) => DType a

instance DType Float

instance DType Double

instance DType Int

instance DType Int32

instance DType Int64

instance DType Word32

instance DType Word64

instance DType Bool

instance (DType a, DType b) => DType (a, b)

instance (DType a, DType b, DType c) => DType (a, b, c)

instance (DType a, DType b, DType c, DType d) => DType (a, b, c, d)

instance (DType a, DType b, DType c, DType d, DType e) => DType (a, b, c, d, e)

instance (DType a, DType b, DType c, DType d, DType e, DType f) => DType (a, b, c, d, e, f)

instance (DType a, DType b, DType c, DType d, DType e, DType f, DType g) => DType (a, b, c, d, e, f, g)

instance (DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h) => DType (a, b, c, d, e, f, g, h)

instance (DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i) => DType (a, b, c, d, e, f, g, h, i)

-- | Embed a host value in an element expression.
element :: forall a. (DType a) => a -> TensorElement a
element = TensorElement . A.constant

-- | A staged tensor expression.
--
-- The constructor and 'unTensor' provide direct access to <https://hackage.haskell.org/package/accelerate Accelerate>.
-- When using this escape hatch, callers must ensure runtime dimensions match the type-level shape.
-- Subsequent Synapse operations rely on this invariant.
newtype Tensor (sh :: [Nat]) a
  = Tensor {unTensor :: A.Acc (A.Array (ShapeToDIM sh) a)}

-- | An element expression evaluated inside a tensor computation.
-- Element expressions describe local scalar arithmetic rather than independently executable tensors.
-- Use a rank-zero t'Tensor' for a scalar differentiation input or result.
newtype TensorElement a
  = TensorElement {unTensorElement :: A.Exp a}

-- | Kind of tensor representations indexed by a static shape and an element type.
-- This synonym describes the type parameters without requiring any tensor operations.
type TensorLike = [Nat] -> Type -> Type

-- | Replace staged tensors with the selected representation, preserving shapes and element types.
-- Use @AG.Tensor s@ inside differentiation and @TensorData@ for materialized values.
-- The resulting type must uniquely identify both the representation and the original staged type.
-- This correspondence lets execution and differentiation infer their ordinary tensor types.
-- Single-constructor tensor records derive @Generic@ and define the correspondence once:
--
-- @
-- type instance WithTensor t (Parameters Tensor a) = Parameters t a
-- @
--
-- Records must have one constructor and contain at least one field, with every field holding a tensor or a nested tensor record.
-- Sum types and records containing ordinary configuration fields are not supported by execution or differentiation.
type family WithTensor (t :: TensorLike) a = (result :: Type) | result -> t a

type instance WithTensor t (Tensor sh a) = t sh a

type instance WithTensor t (a1, a2) = (WithTensor t a1, WithTensor t a2)

type instance WithTensor t (a1, a2, a3) = (WithTensor t a1, WithTensor t a2, WithTensor t a3)

type instance WithTensor t (a1, a2, a3, a4) = (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4)

type instance WithTensor t (a1, a2, a3, a4, a5) = (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4, WithTensor t a5)

type instance WithTensor t (a1, a2, a3, a4, a5, a6) = (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4, WithTensor t a5, WithTensor t a6)

type instance WithTensor t (a1, a2, a3, a4, a5, a6, a7) = (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4, WithTensor t a5, WithTensor t a6, WithTensor t a7)

type instance WithTensor t (a1, a2, a3, a4, a5, a6, a7, a8) = (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4, WithTensor t a5, WithTensor t a6, WithTensor t a7, WithTensor t a8)

type instance WithTensor t (a1, a2, a3, a4, a5, a6, a7, a8, a9) = (WithTensor t a1, WithTensor t a2, WithTensor t a3, WithTensor t a4, WithTensor t a5, WithTensor t a6, WithTensor t a7, WithTensor t a8, WithTensor t a9)

-- | Element expressions associated with a tensor representation.
-- Each element representation identifies its corresponding tensor representation.
type family Element (t :: TensorLike) = (e :: Type -> Type) | e -> t

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
-- Direct construction requires runtime dimensions to match the type-level shape, just as for t'Tensor'.
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
