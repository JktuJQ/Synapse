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
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import qualified Synapse.Tensor.Type as T

deriving instance (Context s, Num (T.Tensor sh a)) => Num (Tensor s sh a)

deriving instance (Context s, Fractional (T.Tensor sh a)) => Fractional (Tensor s sh a)

deriving instance (Context s, Floating (T.Tensor sh a)) => Floating (Tensor s sh a)

deriving instance (Context s, Num (T.TensorElement a)) => Num (TensorElement s a)

deriving instance (Context s, Fractional (T.TensorElement a)) => Fractional (TensorElement s a)

deriving instance (Context s, Floating (T.TensorElement a)) => Floating (TensorElement s a)

instance (Context s, DType a, Num a, Num (T.TensorElement a)) => LinearAlgebra (Tensor s) a where
  dotOp = tensorBinary $ \x y ->
    ( dotOp x y,
      \(T.Tensor gradient) ->
        let scale = T.TensorElement (A.the gradient)
         in (H.map (* scale) y, H.map (* scale) x)
    )
  matmulOp = tensorBinary $ \x y -> (matmulOp x y, matmulPullback x y)

-- | Contract the output gradient with each input over the forward product's free axis.
-- Each contraction uses a generated matrix consumed directly by one reduction.
matmulPullback :: forall lhs rhs out a. (Shape lhs, Shape rhs, Shape out, DType a, Num (T.TensorElement a)) => T.Tensor lhs a -> T.Tensor rhs a -> T.Tensor out a -> (T.Tensor lhs a, T.Tensor rhs a)
matmulPullback x@(T.Tensor left) y@(T.Tensor right) (T.Tensor gradient) =
  case (reverse (shapeOf x), reverse (shapeOf y)) of
    (shared : rows : _, cols : _ : _) ->
      let leftVector = A.flatten left
          rightVector = A.flatten right
          outputVector = A.flatten gradient
          zero = T.unTensorElement (0 :: T.TensorElement a)
          plus :: A.Exp a -> A.Exp a -> A.Exp a
          plus p q = T.unTensorElement (T.TensorElement p + T.TensorElement q)
          times :: A.Exp a -> A.Exp a -> A.Exp a
          times p q = T.unTensorElement (T.TensorElement p * T.TensorElement q)
          leftProducts = A.generate (A.constant (A.Z A.:. shapeSize @lhs A.:. cols)) $ \ix ->
            let linear = A.indexHead (A.indexTail ix)
                col = A.indexHead ix
                rowIndex = linear `A.quot` A.constant shared
                inner = linear `A.rem` A.constant shared
                batch = rowIndex `A.quot` A.constant rows
             in times
                  (outputVector A.! A.index1 (rowIndex * A.constant cols + col))
                  (rightVector A.! A.index1 ((batch * A.constant shared + inner) * A.constant cols + col))
          rightProducts = A.generate (A.constant (A.Z A.:. shapeSize @rhs A.:. rows)) $ \ix ->
            let linear = A.indexHead (A.indexTail ix)
                row = A.indexHead ix
                col = linear `A.rem` A.constant cols
                innerIndex = linear `A.quot` A.constant cols
                batch = innerIndex `A.quot` A.constant shared
                inner = innerIndex `A.rem` A.constant shared
             in times
                  (leftVector A.! A.index1 ((batch * A.constant rows + row) * A.constant shared + inner))
                  (outputVector A.! A.index1 ((batch * A.constant rows + row) * A.constant cols + col))
       in (T.Tensor (A.reshape (A.shape left) (A.fold plus zero leftProducts)), T.Tensor (A.reshape (A.shape right) (A.fold plus zero rightProducts)))
    _ -> error "Synapse.Autograd.matmul: inconsistent input rank"
