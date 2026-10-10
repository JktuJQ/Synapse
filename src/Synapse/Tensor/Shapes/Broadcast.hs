{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Type-level compatibility checks for broadcasting tensors.
module Synapse.Tensor.Shapes.Broadcast
  ( CanBroadcast,
  )
where

import Data.Kind (Constraint)
import GHC.TypeLits (ErrorMessage (ShowType, Text, (:<>:)), Nat, TypeError)
import Synapse.Tensor.Shapes.Dimensions (Reverse)

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
