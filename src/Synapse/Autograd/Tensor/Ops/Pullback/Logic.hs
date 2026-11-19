{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Gradient rules for conditional selection and elementwise extrema.
module Synapse.Autograd.Tensor.Ops.Pullback.Logic
    ( select
    , cond
    , extremum
    ) where

import Synapse.Autograd.Representation (Backpropagatable, zero)

import qualified Synapse.Tensor.Ops.Logic as E

-- | Route the seed using the forward mask, preserving zeros in the inactive branch.
select
    :: (Backpropagatable a, E.Conditional a)
    => E.Condition a
    -> a
    -> a
    -> a
    -> a
    -> (a, a)
select mask _ _ _ gradient = (E.select mask gradient zero, E.select mask zero gradient)

-- | Route the seed to the computation chosen by a scalar predicate.
cond
    :: (Backpropagatable a, E.Conditional a)
    => E.Predicate a
    -> a
    -> a
    -> a
    -> a
    -> (a, a)
cond flag _ _ _ gradient = (E.cond flag gradient zero, E.cond flag zero gradient)

-- | Split equal extrema evenly and zero the contribution of the losing branch.
extremum
    :: (Backpropagatable a, Fractional a, E.Ord a, E.Conditional a, E.Comparison a ~ E.Condition a)
    => (a -> a -> E.Comparison a)
    -> a
    -> a
    -> a
    -> a
    -> (a, a)
extremum loses x y _ gradient =
    let shared = E.select (x E.== y) (gradient / 2) gradient
     in (E.select (loses x y) zero shared, E.select (loses y x) zero shared)
