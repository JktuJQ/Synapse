-- | Public autograd API.
--
-- This module is intended to be imported qualified:
--
-- @
-- import qualified Synapse.Autograd as AG
-- @
module Synapse.Autograd
  ( Backpropagatable (..),
    Context,
    Var (..),
    Tensor (..),
    TensorElement (..),
    Differentiable,
  )
where

import Synapse.Autograd.Differentiable (Differentiable)
import Synapse.Autograd.Tensor (Tensor (..), TensorElement (..))
import Synapse.Autograd.Type (Backpropagatable (..), Context, Var (..))
