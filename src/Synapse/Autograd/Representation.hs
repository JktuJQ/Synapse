{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

{- | Gradient accumulation and differentiation scopes.
'Backpropagatable' supplies gradient identities and accumulation; 'Context' identifies a differentiation scope.
For differentiation itself, use "Synapse.Autograd.Execution".
-}
module Synapse.Autograd.Representation
    ( Context
    , Backpropagatable (..)
    )
where

import qualified Numeric.Backprop as BP

{- | Values supporting gradient accumulation and a default output seed.

The 'Num' defaults are @0@, @1@ and @(+)@; other instances define all three operations explicitly.
For tensors, the type determines the shape of these constants.
'zero' must be a left and right identity for 'add'.
-}
class Backpropagatable a where
    -- | Additive identity for gradient contributions.
    zero :: a
    default zero
        :: Num a
        => a
    zero = 0

    -- | Default output seed used when no explicit output gradient is supplied.
    one :: a
    default one
        :: Num a
        => a
    one = 1

    -- | Combine two gradient contributions for the same value.
    add :: a -> a -> a
    default add
        :: Num a
        => a -> a -> a
    add = (+)

instance Backpropagatable Int
instance Backpropagatable Integer
instance Backpropagatable Float
instance Backpropagatable Double

{- | A differentiation scope supplied by the differentiation entry points.
The parameter @s@ keeps variables from independent scopes separate.
-}
class BP.Reifies s BP.W => Context s

instance BP.Reifies s BP.W => Context s
