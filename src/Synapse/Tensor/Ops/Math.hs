{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Elementwise arithmetic, inner products, and matrix multiplication.
module Synapse.Tensor.Ops.Math (dot, matmul) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import qualified Synapse.Autograd.Tensor.Ops.Pullback.Math as Pullback
import Synapse.Tensor.Ops.Constraints (FloatingDType, FractionalDType, NumDType)
import Synapse.Tensor.Ops.Definition (TensorOps (..), withVJP)
import Synapse.Tensor.Ops.Raw.Construction (full)
import qualified Synapse.Tensor.Ops.Raw.Elementwise as RawElementwise
import qualified Synapse.Tensor.Ops.Raw.Math as Raw
import Synapse.Tensor.Representation (Tensor, TensorLike)
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

-- | Multiply vectors or matrices, broadcasting leading batch dimensions.
matmul
    :: forall {t :: TensorLike} lhs rhs a
     . (TensorOps t a, NumDType a, Shape lhs, Shape rhs, Shape (MatmulShape lhs rhs))
    => t lhs a
    -> t rhs a
    -> t (MatmulShape lhs rhs) a
matmul =
    curry $
        withVJP
            (Proxy @(() :: Constraint))
            (uncurry Raw.matmul)
            (\(left, right) output -> Pullback.matmul left right output)
