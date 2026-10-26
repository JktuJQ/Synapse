{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Variables, differentiation scopes, and gradient accumulation.
module Synapse.Autograd.Variable
  ( Backpropagatable (..),
    Context,
    Var (..),
    stopVariable,
    variableValue,
    liftUnary,
    liftBinary,
  )
where

import qualified Numeric.Backprop as BP
import qualified Numeric.Backprop.Explicit as BPE
import qualified Numeric.Backprop.Internal as BPI

-- | Values whose gradient contributions can be initialized and combined.
-- A 'Num' instance is optional; numeric instances can use the default implementations.
-- Other instances must supply all three operations.
-- The default numeric literals must preserve the shape of the value.
-- Containers whose shape is determined at runtime should override 'zero' and 'one' using the supplied reference value.
class Backpropagatable a where
  -- | Construct a zero gradient with the structure of the reference value.
  zero :: a -> a
  default zero :: (Num a) => a -> a
  zero _ = 0

  -- | Construct the default output gradient with the structure of the reference value.
  -- Numeric and tensor instances use ones, corresponding to differentiation of the sum of the output elements.
  -- Custom instances determine their own default seed.
  one :: a -> a
  default one :: (Num a) => a -> a
  one _ = 1

  -- | Add two gradient contributions for the same value, preserving its structure and shape.
  -- The result of 'zero' must be an additive identity for this operation.
  add :: a -> a -> a
  default add :: (Num a) => a -> a -> a
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
  { -- | Access the underlying <https://hackage.haskell.org/package/backprop backprop> variable without discarding its history.
    unVar :: BP.BVar s a
  }

-- | Read the staged value used to construct nondifferentiable predicates.
variableValue :: Var s a -> a
variableValue = BPI._bvVal . unVar

-- | Preserve a variable's value while discarding its tape edges.
stopVariable :: Var s a -> Var s a
stopVariable = Var . BPE.auto . variableValue

-- | Lift a value computation together with its vector-Jacobian product.
-- The accumulation function combines contributions to the input variable.
liftUnary :: (Context s) => (a -> a -> a) -> (a -> (b, b -> a)) -> Var s a -> Var s b
liftUnary accumulate rule (Var x) = Var (BPE.liftOp1 (BPE.AF accumulate) (BPE.op1 rule) x)

-- | Lift a binary computation with one vector-Jacobian product per input.
liftBinary :: (Context s) => (a -> a -> a) -> (b -> b -> b) -> (a -> b -> (c, c -> (a, b))) -> Var s a -> Var s b -> Var s c
liftBinary accumulateX accumulateY rule (Var x) (Var y) =
  Var (BPE.liftOp2 (BPE.AF accumulateX) (BPE.AF accumulateY) (BPE.op2 rule) x y)
