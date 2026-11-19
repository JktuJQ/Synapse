{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

{- | Differentiable tensor representations and integration with @backprop@.
Constructors and fields preserve the differentiation history of the underlying variables.
-}
module Synapse.Autograd.Tensor.Representation
    ( -- * Tensors
      Tensor (..)
    , WithGrad

      -- * Element expressions
    , TensorElement (..)
    ) where

import GHC.TypeLits (Nat)
import qualified Synapse.Tensor.Representation as T

import Synapse.Autograd.Representation (Backpropagatable)
import Synapse.Tensor.Ops.Constraints (NumDType)
import Synapse.Tensor.Ops.Math ()
import Synapse.Tensor.Structure (WithTensor)

import qualified Numeric.Backprop as BP

{- | A tensor value and its differentiation history in scope @s@.
The shape @sh@ and element type @a@ are those of the underlying 'T.Tensor'.
-}
newtype Tensor s (sh :: [Nat]) a = Tensor
    { unTensor :: BP.BVar s (T.Tensor sh a)
    }

{- | Replace tensor fields with differentiable tensors in scope @s@ using 'WithTensor'.
Shapes, element types and the surrounding structure are preserved.
-}
type WithGrad s a = WithTensor (Tensor s) a

{- | An element expression and its differentiation history in scope @s@.
Differentiation entry points accept tensors rather than standalone element expressions.
-}
newtype TensorElement s a = TensorElement
    { unTensorElement :: BP.BVar s (T.TensorElement a)
    }

instance Num (T.Tensor sh a) => Backpropagatable (T.Tensor sh a)
instance NumDType a => Backpropagatable (T.TensorElement a)
