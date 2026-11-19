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

      -- * Representation correspondence
    , TensorLike

      -- * Tensors
    , Tensor (..)
    , TensorData (..)

      -- * Element expressions and indices
    , TensorElement (..)
    , TensorIndex (..)
    ) where

import Data.Kind (Type)
import GHC.TypeLits (Nat)

import Synapse.Tensor.Shapes.Backend (ShapeToDIM)

import qualified Data.Array.Accelerate as A

{- | Types that can be stored as tensor elements.

'DType' must be supported by the Synapse backend.
-}
class A.Elt a => DType a

instance A.Elt a => DType a

{- | Kind of tensor constructors indexed by a shape and an element type.
Both t'Tensor' and t'TensorData' have this kind.
-}
type TensorLike = [Nat] -> Type -> Type

{- | A staged array computation with shape @sh@ and element type @a@.

For example, @Tensor '[2, 3] Float@ describes a matrix with two rows and three columns.
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
