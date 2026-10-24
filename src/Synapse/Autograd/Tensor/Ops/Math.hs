{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Arithmetic on differentiable tensors and scalar expressions.
module Synapse.Autograd.Tensor.Ops.Math () where

import qualified Data.Array.Accelerate as A
import Synapse.Autograd.Tensor.Helpers (shapeOf, tensorBinary)
import Synapse.Autograd.Tensor.Type (Tensor (..), TensorElement (..))
import Synapse.Autograd.Type (Context)
import Synapse.Tensor.DType (DType)
import qualified Synapse.Tensor.Ops.Elementwise.Helpers as H
import Synapse.Tensor.Ops.Math (LinearAlgebra (..))
import Synapse.Tensor.Ops.Math.Helpers (broadcastBatchIndex, broadcastCopies, broadcastDimensions, contract, matrixDimensions, multiply)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import qualified Synapse.Tensor.Type as T

deriving instance (Context s, Num (T.Tensor sh a)) => Num (Tensor s sh a)

deriving instance (Context s, Fractional (T.Tensor sh a)) => Fractional (Tensor s sh a)

deriving instance (Context s, Floating (T.Tensor sh a)) => Floating (Tensor s sh a)

deriving instance (Context s, Num (T.TensorElement a)) => Num (TensorElement s a)

deriving instance (Context s, Fractional (T.TensorElement a)) => Fractional (TensorElement s a)

deriving instance (Context s, Floating (T.TensorElement a)) => Floating (TensorElement s a)

-- | Contract directly into each original input element, summing all broadcast batch copies.
matmulPullback :: forall lhs rhs out a. (Shape lhs, Shape rhs, Shape out, DType a, Num (T.TensorElement a)) => T.Tensor lhs a -> T.Tensor rhs a -> T.Tensor out a -> (T.Tensor lhs a, T.Tensor rhs a)
matmulPullback x@(T.Tensor left) y@(T.Tensor right) (T.Tensor gradient) =
  ( T.Tensor (A.reshape (A.shape left) leftGradient),
    T.Tensor (A.reshape (A.shape right) rightGradient)
  )
  where
    (leftBatch, rows, shared) = matrixDimensions True (shapeOf x)
    (rightBatch, _, cols) = matrixDimensions False (shapeOf y)
    batchDims = broadcastDimensions leftBatch rightBatch
    (leftCopies, expandLeft) = broadcastCopies leftBatch batchDims
    (rightCopies, expandRight) = broadcastCopies rightBatch batchDims
    leftVector = A.flatten left
    rightVector = A.flatten right
    outputVector = A.flatten gradient
    leftGradient = contract (shapeSize @lhs) (leftCopies * cols) $ \linear entry ->
      let rowIndex = linear `A.quot` A.constant shared
          inner = linear `A.rem` A.constant shared
          row = rowIndex `A.rem` A.constant rows
          originalBatch = rowIndex `A.quot` A.constant rows
          batch = expandLeft originalBatch (entry `A.quot` A.constant cols)
          col = entry `A.rem` A.constant cols
          outputOffset = (batch * A.constant rows + row) * A.constant cols + col
          rightOffset = (broadcastBatchIndex rightBatch batchDims batch * A.constant shared + inner) * A.constant cols + col
       in multiply (outputVector A.! A.index1 outputOffset) (rightVector A.! A.index1 rightOffset)
    rightGradient = contract (shapeSize @rhs) (rightCopies * rows) $ \linear entry ->
      let col = linear `A.rem` A.constant cols
          innerIndex = linear `A.quot` A.constant cols
          inner = innerIndex `A.rem` A.constant shared
          originalBatch = innerIndex `A.quot` A.constant shared
          batch = expandRight originalBatch (entry `A.quot` A.constant rows)
          row = entry `A.rem` A.constant rows
          leftOffset = (broadcastBatchIndex leftBatch batchDims batch * A.constant rows + row) * A.constant shared + inner
          outputOffset = (batch * A.constant rows + row) * A.constant cols + col
       in multiply (leftVector A.! A.index1 leftOffset) (outputVector A.! A.index1 outputOffset)

instance (Context s, DType a, Num a, Num (T.TensorElement a)) => LinearAlgebra (Tensor s) a where
  dotOp = tensorBinary $ \x y ->
    ( dotOp x y,
      \(T.Tensor gradient) ->
        let scale = T.TensorElement (A.the gradient)
         in (H.map (* scale) y, H.map (* scale) x)
    )
  matmulOp = tensorBinary $ \x y -> (matmulOp x y, matmulPullback x y)
