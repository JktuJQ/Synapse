{-# LANGUAGE TypeFamilies #-}

-- | Gradient rules for floating-point scaling and angles.
module Synapse.Autograd.Tensor.Ops.Pullback.Numeric
    ( atan2
    , significand
    , scaleFloat
    ) where

import Prelude hiding (atan2, scaleFloat, significand)

import qualified Synapse.Tensor.Ops.Numeric as E

-- | Differentiate the polar angle away from the origin and its branch cut.
atan2
    :: Fractional a
    => a
    -> a
    -> a
    -> a
    -> (a, a)
atan2 y x _ seed =
    let denominator = x * x + y * y
     in (seed * x / denominator, negate seed * y / denominator)

-- | Scale the seed by the input's locally constant normalization exponent.
significand
    :: (E.RealFloat a, Num (E.ExponentResult a))
    => a
    -> a
    -> a
    -> a
significand input _ = E.scaleFloat (negate $ E.exponent input)

-- | Scale the seed by the same power of the radix as the forward value.
scaleFloat
    :: E.RealFloat a
    => E.ExponentResult a
    -> a
    -> a
    -> a
    -> a
scaleFloat power _ _ = E.scaleFloat power
