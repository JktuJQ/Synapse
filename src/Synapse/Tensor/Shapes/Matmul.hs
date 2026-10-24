{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Output shapes for vector, matrix, and broadcast batched matrix products.
module Synapse.Tensor.Shapes.Matmul (MatmulShape) where

import GHC.TypeLits (ErrorMessage (ShowType, Text, (:<>:)), Nat, TypeError)
import Synapse.Tensor.Shapes.Broadcast (BroadcastShape)
import Synapse.Tensor.Shapes.Dimensions (Reverse)

-- | Append the matrix result axes to the common batch shape.
type family Append (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
  Append '[] rhs = rhs
  Append (dim ': rest) rhs = dim ': Append rest rhs

-- | Validate the contracted dimensions before broadcasting the reversed batch shapes.
type family MatmulDimensions (lhs :: Nat) (rhs :: Nat) (leftBatch :: [Nat]) (rightBatch :: [Nat]) (result :: [Nat]) :: [Nat] where
  MatmulDimensions shared shared leftBatch rightBatch result = Append (BroadcastShape (Reverse leftBatch) (Reverse rightBatch)) result
  MatmulDimensions lhs rhs leftBatch rightBatch result =
    TypeError ('Text "Matmul inner dimensions must match: " ':<>: 'ShowType lhs ':<>: 'Text " and " ':<>: 'ShowType rhs)

-- | Promote vectors for contraction and omit their inserted result axes.
type family MatmulReversed (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
  MatmulReversed '[] rhs = TypeError ('Text "Matmul requires tensors of rank at least one")
  MatmulReversed lhs '[] = TypeError ('Text "Matmul requires tensors of rank at least one")
  MatmulReversed '[lhs] '[rhs] = MatmulDimensions lhs rhs '[] '[] '[]
  MatmulReversed '[lhs] (cols ': rhs ': batch) = MatmulDimensions lhs rhs '[] batch '[cols]
  MatmulReversed (lhs ': rows ': batch) '[rhs] = MatmulDimensions lhs rhs batch '[] '[rows]
  MatmulReversed (lhs ': rows ': leftBatch) (cols ': rhs ': rightBatch) = MatmulDimensions lhs rhs leftBatch rightBatch '[rows, cols]

-- | Matrix product shape with vector promotion and broadcasting of leading batch axes.
-- Two vectors return a scalar; scalar operands and incompatible dimensions are rejected.
type MatmulShape lhs rhs = MatmulReversed (Reverse lhs) (Reverse rhs)
