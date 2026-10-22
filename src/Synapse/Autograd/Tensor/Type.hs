{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Differentiable adapters for staged tensors and element expressions.
module Synapse.Autograd.Tensor.Type
  ( Tensor (..),
    TensorElement (..),
  )
where

import GHC.TypeLits (Nat)
import Synapse.Autograd.Differentiable (Differentiable)
import Synapse.Autograd.Type (Backpropagatable, Var)
import Synapse.Tensor.Ops.Math ()
import qualified Synapse.Tensor.Type as T

-- | A differentiable tensor with statically known dimensions.
-- The constructor and 'unTensor' preserve differentiation history when converting to and from the underlying variable.
newtype Tensor s (sh :: [Nat]) a = Tensor
  { unTensor :: Var s (T.Tensor sh a)
  }

-- | A differentiable scalar expression used inside tensor computations.
-- The constructor and 'unTensorElement' preserve differentiation history when converting to and from the underlying variable.
newtype TensorElement s a = TensorElement
  { unTensorElement :: Var s (T.TensorElement a)
  }

instance (Num (T.Tensor sh a)) => Backpropagatable (T.Tensor sh a)

instance (Num (T.TensorElement a)) => Backpropagatable (T.TensorElement a)

type instance Differentiable s (T.Tensor sh a) = Tensor s sh a

type instance Differentiable s (T.TensorElement a) = TensorElement s a

type instance T.Element (Tensor s) = TensorElement s
