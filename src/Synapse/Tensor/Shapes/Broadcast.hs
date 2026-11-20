{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Type-level broadcasting of tensor shapes.
module Synapse.Tensor.Shapes.Broadcast
    ( CanBroadcast
    , BroadcastShape
    )
where

import Data.Kind (Constraint)
import GHC.TypeLits (ErrorMessage (ShowType, Text, (:<>:)), Nat, TypeError)

import Synapse.Tensor.Shapes.Backend (Reverse)

-- Common broadcast shape

-- | Combine equal or singleton dimensions; pairing zero with one produces zero.
type family BroadcastDim (lhs :: Nat) (rhs :: Nat) :: Nat where
    BroadcastDim 1 rhs = rhs
    BroadcastDim lhs 1 = lhs
    BroadcastDim dim dim = dim
    BroadcastDim lhs rhs =
        TypeError ('Text "Cannot broadcast dimensions " ':<>: 'ShowType lhs ':<>: 'Text " and " ':<>: 'ShowType rhs)

-- | Combine reversed shapes so that dimensions are aligned from the trailing axis.
type family BroadcastReversed (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
    BroadcastReversed '[] rhs = rhs
    BroadcastReversed lhs '[] = lhs
    BroadcastReversed (lhs ': left) (rhs ': right) = BroadcastDim lhs rhs ': BroadcastReversed left right

{- | Common shape obtained by broadcasting two shapes together.

Trailing dimensions must be equal or include a singleton; unmatched leading dimensions are retained.
@BroadcastShape '[2, 1, 4] '[3, 4]@ is @'[2, 3, 4]@.
A scalar broadcasts with any shape; incompatible dimensions produce a type error.
-}
type BroadcastShape lhs rhs = Reverse (BroadcastReversed (Reverse lhs) (Reverse rhs))

-- Directed broadcast compatibility

-- | Require a source dimension to equal its target dimension or have size one.
type family CanBroadcastDim (fromDim :: Nat) (toDim :: Nat) :: Constraint where
    CanBroadcastDim 1 toDim = ()
    CanBroadcastDim dim dim = ()
    CanBroadcastDim fromDim toDim =
        TypeError
            ( 'Text "Cannot broadcast dimension "
                ':<>: 'ShowType fromDim
                ':<>: 'Text " to "
                ':<>: 'ShowType toDim
            )

-- | Check directed broadcasting of reversed shapes, rejecting extra source dimensions.
type family BroadcastSuffix (from :: [Nat]) (to :: [Nat]) :: Constraint where
    BroadcastSuffix '[] _ = ()
    BroadcastSuffix (fromDim ': fromRest) '[] =
        TypeError
            ( 'Text "Cannot broadcast source shape with extra leading dimension "
                ':<>: 'ShowType fromDim
            )
    BroadcastSuffix (fromDim ': fromRest) (toDim ': toRest) =
        (CanBroadcastDim fromDim toDim, BroadcastSuffix fromRest toRest)

{- | Require that a source shape can be expanded to a target shape.

The source rank must not exceed the target rank.
Aligned trailing dimensions must be equal or the source dimension must be one.
For example, @CanBroadcast '[3] '[2, 3]@ holds, but the reverse direction does not.
The scalar shape can broadcast to any target, including an empty tensor.
-}
type CanBroadcast from to = BroadcastSuffix (Reverse from) (Reverse to)
