{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Pure parameter updates with explicit optimizer state.
-- Updates build tensor expressions without executing them.
-- Parameters and gradients must have corresponding fields and tensor shapes.
module Synapse.NN.Optimizers.Type
  ( Optimizer (..),
    updateParameters,
  )
where

import Control.Monad.Trans.State.Strict (State)
import Synapse.Autograd.Execution.Variable (ValueTree (..), Values (..))
import Synapse.Autograd.Tensor ()
import Synapse.Autograd.Type (Differentiable, ParametersConstraints)
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import qualified Synapse.Tensor.Type as T

-- | Optimizers for a parameter structure.
-- The initial state is constructed before the first step.
-- State actions compose successive updates without manually passing optimizer state.
-- Define 'initialState' and 'optimizerStep'; run steps with @runState@ or @evalState@.
class Optimizer optimizer parameters where
  -- | Accumulated values carried between updates.
  type OptimizerState optimizer parameters

  -- | Initialize optimizer state for the supplied parameters.
  initialState :: optimizer -> parameters -> OptimizerState optimizer parameters

  -- | Update parameters using their gradients in a pure state computation.
  -- Initialize the state once with 'initialState' and carry it between updates.
  -- Successive steps compose directly in @do@ notation.
  optimizerStep :: optimizer -> parameters -> parameters -> State (OptimizerState optimizer parameters) parameters

-- | Compute new parameters by applying an update formula to each parameter and its corresponding gradient.
-- The formula receives the current parameter component first and its gradient second, and returns the updated component.
-- The parameter and gradient structures must correspond, including the shapes of their tensor components.
-- Select the required component operations with a type application such as @\@Num@, @\@Fractional@, or @\@Floating@.
-- The containing records need no numeric instances.
--
-- @
-- updateParameters \@Fractional (\parameter gradient -> parameter - 0.01 * gradient)
--   parameters gradients
-- @
updateParameters ::
  forall c parameters.
  (Differentiable parameters, ParametersConstraints c parameters) =>
  -- | Update formula taking the current parameter component first and its gradient second.
  (forall a. (c a) => a -> a -> a) ->
  -- | Current parameters.
  parameters ->
  -- | Corresponding gradients.
  parameters ->
  -- | Updated parameters with the same structure and tensor shapes.
  parameters
updateParameters formula =
  let applyUpdate :: ValueTree c p -> p -> p -> p
      applyUpdate TensorValues = formula
      applyUpdate (PairValues left right) = \(p1, p2) (g1, g2) ->
        (applyUpdate left p1 g1, applyUpdate right p2 g2)
   in \parameters gradients -> rebuild @T.Tensor (applyUpdate (values @(Leaves T.Tensor parameters) @c) (flatten @T.Tensor parameters) (flatten @T.Tensor gradients))
