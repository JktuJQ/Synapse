{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Variables, differentiation scopes, and gradient accumulation.
module Synapse.Autograd.Type
  ( Backpropagatable (..),
    Context,
    Var (..),
  )
where

import qualified Numeric.Backprop as BP

-- | Numeric values whose gradient contributions can be initialized and combined.
-- The defaults use numeric literals, which must preserve the shape of the value.
-- Containers whose shape is determined at runtime should override 'zero' and 'one' using the supplied reference value.
class (Num a) => Backpropagatable a where
  -- | Construct a zero gradient with the structure of the reference value.
  zero :: a -> a
  zero _ = 0

  -- | Construct an initial gradient of ones with the structure of the reference value.
  -- For nonscalar results, this seed corresponds to differentiating the sum of their components.
  one :: a -> a
  one _ = 1

  -- | Add two gradient contributions for the same value.
  add :: a -> a -> a
  add = (+)

instance Backpropagatable Int

instance Backpropagatable Integer

instance Backpropagatable Float

instance Backpropagatable Double

-- | A scope that owns the differentiation tape.
-- The scope parameter prevents variables from independent runs from being mixed.
-- Differentiation entry points supply this context; callers do not construct it.
class (BP.Reifies s BP.W) => Context s

instance (BP.Reifies s BP.W) => Context s

-- | A value participating in a differentiation scope.
-- The constructor and 'unVar' are escape hatches for custom operations using the underlying <https://hackage.haskell.org/package/backprop backprop> library.
newtype Var s a = Var
  { unVar :: BP.BVar s a
  }
