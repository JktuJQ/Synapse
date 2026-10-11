{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Output shapes and compatibility checks for batched matrix multiplication.
module Synapse.Tensor.Shapes.Matmul (MatmulShape) where

import GHC.TypeLits (Nat, TypeError, ErrorMessage (Text, ShowType, (:<>:)))
import Synapse.Tensor.Shapes.Dimensions (Reverse)

type family MatmulBatches (lhs :: [Nat]) (rhs :: [Nat]) (rows :: Nat) (cols :: Nat) :: [Nat] where
  MatmulBatches batch batch rows cols = Reverse (cols ': rows ': batch)
  MatmulBatches lhs rhs rows cols =
    TypeError ('Text "Matmul batch shapes must match: " ':<>: 'ShowType (Reverse lhs) ':<>: 'Text " and " ':<>: 'ShowType (Reverse rhs))

type family MatmulDimensions (lhs :: Nat) (rhs :: Nat) (rows :: Nat) (cols :: Nat) (leftBatch :: [Nat]) (rightBatch :: [Nat]) :: [Nat] where
  MatmulDimensions shared shared rows cols leftBatch rightBatch = MatmulBatches leftBatch rightBatch rows cols
  MatmulDimensions lhs rhs rows cols leftBatch rightBatch =
    TypeError ('Text "Matmul inner dimensions must match: " ':<>: 'ShowType lhs ':<>: 'Text " and " ':<>: 'ShowType rhs)

type family MatmulReversed (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
  MatmulReversed (shared ': rows ': leftBatch) (cols ': sharedRight ': rightBatch) =
    MatmulDimensions shared sharedRight rows cols leftBatch rightBatch
  MatmulReversed lhs rhs =
    TypeError ('Text "Matmul requires tensors of rank at least two, got " ':<>: 'ShowType (Reverse lhs) ':<>: 'Text " and " ':<>: 'ShowType (Reverse rhs))

-- | Multiply the last two dimensions and preserve exactly matching leading batch dimensions.
-- Rank and compatibility errors are reported while computing the output shape.
type MatmulShape lhs rhs = MatmulReversed (Reverse lhs) (Reverse rhs)
