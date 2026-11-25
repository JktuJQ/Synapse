{-# LANGUAGE CPP #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Elementwise arithmetic, inner products, and matrix multiplication.
module Synapse.Tensor.Ops.Math (dot, matmul, matmult, tmatmul, tmatmult) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import Synapse.Autograd.Tensor.Ops.Pullback.Math qualified as Pullback
import Synapse.Tensor.Ops.Constraints (FloatingDType, FractionalDType, NumDType)
import Synapse.Tensor.Ops.Definition (TensorOps (..), withVJP)
import Synapse.Tensor.Ops.Raw.Construction (full)
import Synapse.Tensor.Ops.Raw.Elementwise qualified as RawElementwise
import Synapse.Tensor.Ops.Raw.Math qualified as Raw
import Synapse.Tensor.Ops.Shape (transpose)
import Synapse.Tensor.Representation (Tensor, TensorLike)
import Synapse.Tensor.Shapes.Axes (TransposeShape)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Matmul (MatmulShape)

instance (Shape sh, NumDType a) => Num (Tensor sh a) where
    (+) = RawElementwise.zipWith (+)
    (-) = RawElementwise.zipWith (-)
    (*) = RawElementwise.zipWith (*)
    negate = RawElementwise.map negate
    abs = RawElementwise.map abs
    signum = RawElementwise.map signum
    fromInteger = full @sh . fromInteger
instance (Shape sh, FractionalDType a) => Fractional (Tensor sh a) where
    (/) = RawElementwise.zipWith (/)
    recip = RawElementwise.map recip
    fromRational = full @sh . fromRational
instance (Shape sh, FloatingDType a) => Floating (Tensor sh a) where
    pi = full @sh pi
    exp = RawElementwise.map exp
    log = RawElementwise.map log
    sqrt = RawElementwise.map sqrt
    (**) = RawElementwise.zipWith (**)
    logBase = RawElementwise.zipWith logBase
    sin = RawElementwise.map sin
    cos = RawElementwise.map cos
    tan = RawElementwise.map tan
    asin = RawElementwise.map asin
    acos = RawElementwise.map acos
    atan = RawElementwise.map atan
    sinh = RawElementwise.map sinh
    cosh = RawElementwise.map cosh
    tanh = RawElementwise.map tanh
    asinh = RawElementwise.map asinh
    acosh = RawElementwise.map acosh
    atanh = RawElementwise.map atanh

-- | Sum pairwise products of equally sized vectors.
dot
    :: forall {t :: TensorLike} n a
     . (TensorOps t a, NumDType a, Shape '[n])
    => t '[n] a
    -> t '[n] a
    -> t '[] a
dot =
    curry $
        withVJP
            (Proxy @(() :: Constraint))
            (uncurry Raw.dot)
            (\(left, right) output -> Pullback.dot left right output)

-- | Apply a matrix product and its pullback in the selected operand orientations.
matmulWith
    :: (TensorOps t a, NumDType a, Shape lhs, Shape rhs, Shape out)
    => Bool
    -> Bool
    -> t lhs a
    -> t rhs a
    -> t out a
matmulWith transposeLeft transposeRight =
    curry $
        withVJP
            (Proxy @(() :: Constraint))
            (uncurry $ Raw.matmulWith transposeLeft transposeRight)
            (\(left, right) -> Pullback.matmulWith transposeLeft transposeRight left right)

-- | Multiply vectors or matrices, broadcasting leading batch dimensions.
matmul
    :: forall {t :: TensorLike} lhs rhs a
     . (TensorOps t a, NumDType a, Shape lhs, Shape rhs, Shape (MatmulShape lhs rhs))
    => t lhs a
    -> t rhs a
    -> t (MatmulShape lhs rhs) a
matmul = matmulWith False False

-- | Multiply by the transpose of the right operand.
matmult
    :: forall {t :: TensorLike} lhs rhs a
     . ( TensorOps t a
       , NumDType a
       , Shape lhs
       , Shape rhs
       , Shape (TransposeShape rhs)
       , Shape (MatmulShape lhs (TransposeShape rhs))
       )
    => t lhs a
    -> t rhs a
    -> t (MatmulShape lhs (TransposeShape rhs)) a
#ifdef SYNAPSE_BLAS
matmult = matmulWith False True
#else
matmult left right = matmul left (transpose right)
#endif

-- | Multiply the transpose of the left operand by the right operand.
tmatmul
    :: forall {t :: TensorLike} lhs rhs a
     . ( TensorOps t a
       , NumDType a
       , Shape lhs
       , Shape (TransposeShape lhs)
       , Shape rhs
       , Shape (MatmulShape (TransposeShape lhs) rhs)
       )
    => t lhs a
    -> t rhs a
    -> t (MatmulShape (TransposeShape lhs) rhs) a
#ifdef SYNAPSE_BLAS
tmatmul = matmulWith True False
#else
tmatmul left right = matmul (transpose left) right
#endif

-- | Multiply the transposes of both operands.
tmatmult
    :: forall {t :: TensorLike} lhs rhs a
     . ( TensorOps t a
       , NumDType a
       , Shape lhs
       , Shape rhs
       , Shape (TransposeShape lhs)
       , Shape (TransposeShape rhs)
       , Shape (MatmulShape (TransposeShape lhs) (TransposeShape rhs))
       )
    => t lhs a
    -> t rhs a
    -> t (MatmulShape (TransposeShape lhs) (TransposeShape rhs)) a
#ifdef SYNAPSE_BLAS
tmatmult = matmulWith True True
#else
tmatmult left right = matmul (transpose left) (transpose right)
#endif

#ifdef SYNAPSE_BLAS
{-# INLINE [2] matmul #-}
{-# INLINE [2] matmult #-}
{-# INLINE [2] tmatmul #-}
{-# INLINE [1] matmulWith #-}

{-# RULES
"matmul/leftTranspose" [~1] forall t lhs rhs a out. forall (left :: t lhs a) (right :: t rhs a).
    matmulWith False False (transpose left :: t out a) right = matmulWith True False left right
"matmul/rightTranspose" [~1] forall t lhs rhs a out. forall (left :: t lhs a) (right :: t rhs a).
    matmulWith False False left (transpose right :: t out a) = matmulWith False True left right
"matmult/leftTranspose" [~1] forall t lhs rhs a out. forall (left :: t lhs a) (right :: t rhs a).
    matmulWith False True (transpose left :: t out a) right = matmulWith True True left right
"tmatmul/rightTranspose" [~1] forall t lhs rhs a out. forall (left :: t lhs a) (right :: t rhs a).
    matmulWith True False left (transpose right :: t out a) = matmulWith True True left right
  #-}
#endif
