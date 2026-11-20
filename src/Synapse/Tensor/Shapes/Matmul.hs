{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Output shapes for vector, matrix, and broadcast batched matrix products.
module Synapse.Tensor.Shapes.Matmul (MatmulShape) where

import GHC.TypeLits (ErrorMessage (ShowType, Text, (:<>:)), Nat, TypeError)

import Synapse.Tensor.Shapes.Backend (Reverse)
import Synapse.Tensor.Shapes.Broadcast (BroadcastShape)

-- | Append the matrix result dimensions to the common broadcast batch shape.
type family Append (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
    Append '[] rhs = rhs
    Append (dim ': rest) rhs = dim ': Append rest rhs

-- | Require matching contraction dimensions, then broadcast the reversed batch shapes.
type family
    MatmulDimensions (lhs :: Nat) (rhs :: Nat) (leftBatch :: [Nat]) (rightBatch :: [Nat]) (result :: [Nat])
        :: [Nat]
    where
    MatmulDimensions shared shared leftBatch rightBatch result =
        Append (BroadcastShape (Reverse leftBatch) (Reverse rightBatch)) result
    MatmulDimensions lhs rhs leftBatch rightBatch result =
        TypeError
            ('Text "Matmul inner dimensions must match: " ':<>: 'ShowType lhs ':<>: 'Text " and " ':<>: 'ShowType rhs)

-- | Determine contraction and result dimensions from reversed shapes, promoting vectors as needed.
type family MatmulReversed (lhs :: [Nat]) (rhs :: [Nat]) :: [Nat] where
    MatmulReversed '[] rhs = TypeError ('Text "Matmul requires tensors of rank at least one")
    MatmulReversed lhs '[] = TypeError ('Text "Matmul requires tensors of rank at least one")
    MatmulReversed '[lhs] '[rhs] = MatmulDimensions lhs rhs '[] '[] '[]
    MatmulReversed '[lhs] (cols ': rhs ': batch) = MatmulDimensions lhs rhs '[] batch '[cols]
    MatmulReversed (lhs ': rows ': batch) '[rhs] = MatmulDimensions lhs rhs batch '[] '[rows]
    MatmulReversed (lhs ': rows ': leftBatch) (cols ': rhs ': rightBatch) =
        MatmulDimensions lhs rhs leftBatch rightBatch '[rows, cols]

{- | Matrix product shape with vector promotion and broadcasting of leading batch dimensions.

Two vectors of equal length produce a scalar.
A matrix times a vector retains its row dimension; a vector times a matrix retains its column dimension.
For two matrices, the left column count must equal the right row count.
Batch dimensions broadcast independently of the matrix dimensions.

@
MatmulShape '[4] '[4]          = '[]
MatmulShape '[2, 4] '[4]       = '[2]
MatmulShape '[4] '[4, 3]       = '[3]
MatmulShape '[5, 2, 4] '[4, 3] = '[5, 2, 3]
@

Scalar operands, mismatched contraction dimensions, and incompatible batch dimensions produce type errors.
-}
type MatmulShape lhs rhs = MatmulReversed (Reverse lhs) (Reverse rhs)
