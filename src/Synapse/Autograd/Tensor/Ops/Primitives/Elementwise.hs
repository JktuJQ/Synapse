{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Gradient rules for conditional selection and elementwise extrema.
module Synapse.Autograd.Tensor.Ops.Primitives.Elementwise (select, cond, extremum) where

import Synapse.Autograd.Variable (Backpropagatable, zero)
import qualified Synapse.Tensor.Ops.Elementwise as E

-- | Route the seed using the forward mask, preserving zeros in the inactive branch.
select :: (Backpropagatable a, E.Conditional a) => E.Condition a -> a -> a -> a -> a -> (a, a)
select mask x y _ gradient = (E.select mask gradient (zero x), E.select mask (zero y) gradient)

-- | Route the seed to the computation chosen by a scalar predicate.
cond :: (Backpropagatable a, E.Conditional a) => E.Predicate a -> a -> a -> a -> a -> (a, a)
cond flag x y _ gradient = (E.cond flag gradient (zero x), E.cond flag (zero y) gradient)

-- | Split equal extrema evenly and zero the contribution of the losing branch.
extremum :: (Backpropagatable a, Fractional a, E.Ord a, E.Conditional a, E.Comparison a ~ E.Condition a) => (a -> a -> E.Comparison a) -> a -> a -> a -> a -> (a, a)
extremum loses x y _ gradient =
  let shared = E.select (x E.== y) (gradient / 2) gradient
   in (E.select (loses x y) (zero x) shared, E.select (loses y x) (zero y) shared)
