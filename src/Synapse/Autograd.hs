-- | Reverse-mode differentiation of ordinary values and staged tensor expressions.
-- 'WithGrad' selects the representation used inside a differentiable function: t'Var' for scalar numbers and t'Tensor' for tensors.
-- 'backprop' returns the function's value and its input gradient, while 'gradient' returns only the gradient.
-- 'pullback' accepts a caller-supplied output gradient instead of the default seed.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.Autograd as AG
--
-- -- Returns (9, 6).
-- AG.backprop (\x -> x * x) (3 :: Double)
-- @
--
-- Tensor values and gradients remain staged expressions; execute them through "Synapse.Tensor".
-- Differentiable tensor operations use the same polymorphic functions as ordinary tensor operations.
-- See "Synapse.Autograd.Tensor" for the supported operations and "Synapse.Autograd.Differentiable" for custom records.
-- This API provides differentiation primitives rather than layers, optimizers, or a training loop.
module Synapse.Autograd
  ( Backpropagatable (..),
    Context,
    Var (..),
    Tensor (..),
    TensorElement (..),
    Differentiable (WithGrad),
    evaluate,
    backprop,
    gradient,
    pullback,
    stopGrad,
    constant,
  )
where

import Synapse.Autograd.Differentiable (Differentiable (WithGrad))
import Synapse.Autograd.Execution (backprop, constant, evaluate, gradient, pullback, stopGrad)
import Synapse.Autograd.Tensor (Tensor (..), TensorElement (..))
import Synapse.Autograd.Type (Backpropagatable (..), Context, Var (..))
