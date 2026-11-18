{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Tensor representations, element expressions, and indices.
module Synapse.Tensor.Representation
    ( -- * Element types
      DType
    , Float32
    , Float64

      -- * Tensors
    , Tensor (..)
    , TensorData (..)

      -- * Element expressions and indices
    , TensorElement (..)
    , TensorIndex (..)

      -- * Representation correspondence
    , TensorLike
    ) where

import Data.Int (Int32, Int64)
import Data.Kind (Type)
import Data.Word (Word32, Word64)
import GHC.TypeLits (Nat)

import Synapse.Tensor.Shapes.Backend (ShapeToDIM)

import qualified Data.Array.Accelerate as A

-- | 32-bit floating-point element type; an alias for 'Float'.
type Float32 = Float

-- | 64-bit floating-point element type; an alias for 'Double'.
type Float64 = Double

{- | Types that can be stored as tensor elements.

An instance requires an Accelerate 'A.Elt' representation.
Numeric and comparison operations impose additional constraints on the element type.
Tuple instances describe tuple-valued elements.
-}
class A.Elt a => DType a

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
instance
    (DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h)
    => DType (a, b, c, d, e, f, g, h)
instance
    (DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i)
    => DType (a, b, c, d, e, f, g, h, i)

{- | Kind of tensor constructors indexed by a shape and an element type.
Both t'Tensor' and t'TensorData' have this kind.
-}
type TensorLike = [Nat] -> Type -> Type

{- | A staged array computation with shape @sh@ and element type @a@.

For example, @Tensor '[2, 3] Float32@ describes a matrix with two rows and three columns.
The empty shape @'[]@ describes a scalar array.
Operations build a computation graph; backend execution produces t'TensorData'.

Direct construction must preserve the dimensions specified by @sh@.
Subsequent operations assume this invariant without checking it.
-}
newtype Tensor (sh :: [Nat]) a = Tensor
    { unTensor :: A.Acc (A.Array (ShapeToDIM sh) a)
    }

deriving stock instance Show (A.Acc (A.Array (ShapeToDIM sh) a)) => Show (Tensor sh a)

{- | A materialized array with shape @sh@ and element type @a@.

Backend computations accept and return values in this representation.
Direct construction must preserve the dimensions specified by @sh@.
-}
newtype TensorData (sh :: [Nat]) a = TensorData
    { unTensorData :: A.Array (ShapeToDIM sh) a
    }

deriving stock instance Show (A.Array (ShapeToDIM sh) a) => Show (TensorData sh a)

{- | An element expression within a staged array computation.

Expressions may depend on the current element or its coordinates.
Arithmetic instances construct scalar expressions in the computation graph.
Use a rank-zero t'Tensor' for an independently executable scalar computation.
-}
newtype TensorElement a = TensorElement
    { unTensorElement :: A.Exp a
    }

deriving stock instance Show (A.Exp a) => Show (TensorElement a)
deriving newtype instance Num (A.Exp a) => Num (TensorElement a)
deriving newtype instance Fractional (A.Exp a) => Fractional (TensorElement a)
deriving newtype instance Floating (A.Exp a) => Floating (TensorElement a)

{- | A complete multidimensional index for shape @sh@.

Each coordinate corresponds to one axis of the shape.
Tensor generators supply indices within their output bounds.
Direct construction does not check coordinate bounds;
callers must ensure they are valid for the indexed array.
-}
newtype TensorIndex (sh :: [Nat]) = TensorIndex
    { unTensorIndex :: A.Exp (ShapeToDIM sh)
    }

deriving stock instance Show (A.Exp (ShapeToDIM sh)) => Show (TensorIndex sh)
