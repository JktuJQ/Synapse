-- | Reverse-mode differentiation of staged tensor functions.
-- Tensor records derive @Generic@ and define one @WithTensor@ correspondence in "Synapse.Tensor" for differentiation and execution.
-- Each tensor is registered independently; results and gradients retain their ordinary tensor structure.
-- Scalars use rank-zero tensors, while element expressions are implementation details of tensor operations.
-- Tensor primitives require fractional element types when recording gradients.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.Autograd as AG
-- import qualified Synapse.Tensor as T
--
-- AG.backprop (\x -> x * x) (T.full @'[] (3 :: Double))
-- @
--
-- Tensor values and gradients remain staged; execute them through "Synapse.Tensor".
module Synapse.Autograd
  ( Context,
    Var (..),
    Tensor (..),
    Differentiable,
    WithGrad,
    ParametersConstraints,
    evaluate,
    backprop,
    gradient,
    pullback,
    stopGrad,
    constant,
  )
where

import Synapse.Autograd.Execution (backprop, constant, evaluate, gradient, pullback, stopGrad)
import Synapse.Autograd.Tensor (Tensor (..))
import Synapse.Autograd.Type (Context, Differentiable, ParametersConstraints, Var (..), WithGrad)
