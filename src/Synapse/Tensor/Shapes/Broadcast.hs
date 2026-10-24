{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Type-level compatibility checks for broadcasting tensors.
module Synapse.Tensor.Shapes.Broadcast
  ( CanBroadcast,
    BroadcastShape,
  )
where

import Data.Kind (Constraint)
import GHC.TypeLits (ErrorMessage (ShowType, Text, (:<>:)), Nat, TypeError)
import Synapse.Tensor.Shapes.Dimensions (Reverse)

-- | Combine one pair of broadcast dimensions, including empty dimensions paired with one.
type family BroadcastDim (lhs :: Nat) (rhs :: Nat) :: Nat where
  BroadcastDim 1 rhs = rhs
  BroadcastDim lhs 1 = lhs
  BroadcastDim dim dim = dim
  BroadcastDim lhs rhs = TypeError ('Text "Cannot broadcast dimensions " ':<>: 'ShowType lhs ':<>: 'Text " and " ':<>: 'ShowType rhs)

-- | Combine reversed shapes so their trailing dimensions align.
type family BroadcastReversed (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
  BroadcastReversed '[] rhs = rhs
  BroadcastReversed lhs '[] = lhs
  BroadcastReversed (lhs ': left) (rhs ': right) = BroadcastDim lhs rhs ': BroadcastReversed left right

-- | Common shape obtained by broadcasting two shapes together.
-- Trailing axes align and each pair must be equal or contain a singleton dimension.
type BroadcastShape lhs rhs = Reverse (BroadcastReversed (Reverse lhs) (Reverse rhs))

-- | Check whether a single source dimension can broadcast to a target dimension.
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

-- | Compare reversed shapes dimension-by-dimension for broadcast compatibility.
type family BroadcastSuffix (from :: [Nat]) (to :: [Nat]) :: Constraint where
  BroadcastSuffix '[] _ = ()
  BroadcastSuffix (fromDim ': fromRest) '[] =
    TypeError
      ( 'Text "Cannot broadcast source shape with extra leading dimension "
          ':<>: 'ShowType fromDim
      )
  BroadcastSuffix (fromDim ': fromRest) (toDim ': toRest) =
    (CanBroadcastDim fromDim toDim, BroadcastSuffix fromRest toRest)

-- | Constraint proving that one shape can be broadcast to another.
--
-- Broadcast compatibility is checked from trailing dimensions, matching NumPy and PyTorch semantics.
type CanBroadcast from to = BroadcastSuffix (Reverse from) (Reverse to)
