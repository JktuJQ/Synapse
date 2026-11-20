{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Arithmetic instances for differentiable tensors and element expressions.
module Synapse.Autograd.Tensor.Ops.Math () where

import Synapse.Autograd.Representation (Context, zero)
import Synapse.Autograd.Tensor.Ops (binaryVJP, unaryVJP)
import Synapse.Autograd.Tensor.Representation (Tensor (..), TensorElement (..))
import Synapse.Tensor.Ops.Constraints (FloatingDType, FractionalDType, NumDType)
import Synapse.Tensor.Shapes.Dimensions (Shape)

import qualified Numeric.Backprop.Explicit as BPE

import qualified Synapse.Autograd.Tensor.Ops.Pullback.Math as Pullback

instance (Context s, Shape sh, NumDType a) => Num (Tensor s sh a) where
    (+) (Tensor x) (Tensor y) = Tensor $ binaryVJP (+) Pullback.add x y
    (-) (Tensor x) (Tensor y) = Tensor $ binaryVJP (-) Pullback.subtract x y
    (*) (Tensor x) (Tensor y) = Tensor $ binaryVJP (*) Pullback.multiply x y
    negate (Tensor x) = Tensor $ unaryVJP negate Pullback.negate x
    abs (Tensor x) = Tensor $ unaryVJP abs Pullback.abs x
    signum (Tensor x) = Tensor $ unaryVJP signum (Pullback.signum zero) x
    fromInteger = Tensor . BPE.auto . fromInteger
instance
    (Context s, Shape sh, FractionalDType a)
    => Fractional (Tensor s sh a)
    where
    (/) (Tensor x) (Tensor y) = Tensor $ binaryVJP (/) Pullback.divide x y
    recip (Tensor x) = Tensor $ unaryVJP recip Pullback.recip x
    fromRational = Tensor . BPE.auto . fromRational
instance
    (Context s, Shape sh, FloatingDType a)
    => Floating (Tensor s sh a)
    where
    pi = Tensor $ BPE.auto pi
    exp (Tensor x) = Tensor $ unaryVJP exp Pullback.exp x
    log (Tensor x) = Tensor $ unaryVJP log Pullback.log x
    sqrt (Tensor x) = Tensor $ unaryVJP sqrt Pullback.sqrt x
    sin (Tensor x) = Tensor $ unaryVJP sin Pullback.sin x
    cos (Tensor x) = Tensor $ unaryVJP cos Pullback.cos x
    tan (Tensor x) = Tensor $ unaryVJP tan Pullback.tan x
    asin (Tensor x) = Tensor $ unaryVJP asin Pullback.asin x
    acos (Tensor x) = Tensor $ unaryVJP acos Pullback.acos x
    atan (Tensor x) = Tensor $ unaryVJP atan Pullback.atan x
    sinh (Tensor x) = Tensor $ unaryVJP sinh Pullback.sinh x
    cosh (Tensor x) = Tensor $ unaryVJP cosh Pullback.cosh x
    tanh (Tensor x) = Tensor $ unaryVJP tanh Pullback.tanh x
    asinh (Tensor x) = Tensor $ unaryVJP asinh Pullback.asinh x
    acosh (Tensor x) = Tensor $ unaryVJP acosh Pullback.acosh x
    atanh (Tensor x) = Tensor $ unaryVJP atanh Pullback.atanh x
    (**) (Tensor x) (Tensor y) = Tensor $ binaryVJP (**) Pullback.power x y
    logBase x y = log y / log x
instance (Context s, NumDType a) => Num (TensorElement s a) where
    (+) (TensorElement x) (TensorElement y) = TensorElement $ binaryVJP (+) Pullback.add x y
    (-) (TensorElement x) (TensorElement y) = TensorElement $ binaryVJP (-) Pullback.subtract x y
    (*) (TensorElement x) (TensorElement y) = TensorElement $ binaryVJP (*) Pullback.multiply x y
    negate (TensorElement x) = TensorElement $ unaryVJP negate Pullback.negate x
    abs (TensorElement x) = TensorElement $ unaryVJP abs Pullback.abs x
    signum (TensorElement x) = TensorElement $ unaryVJP signum (Pullback.signum zero) x
    fromInteger = TensorElement . BPE.auto . fromInteger
instance (Context s, FractionalDType a) => Fractional (TensorElement s a) where
    (/) (TensorElement x) (TensorElement y) = TensorElement $ binaryVJP (/) Pullback.divide x y
    recip (TensorElement x) = TensorElement $ unaryVJP recip Pullback.recip x
    fromRational = TensorElement . BPE.auto . fromRational
instance (Context s, FloatingDType a) => Floating (TensorElement s a) where
    pi = TensorElement $ BPE.auto pi
    exp (TensorElement x) = TensorElement $ unaryVJP exp Pullback.exp x
    log (TensorElement x) = TensorElement $ unaryVJP log Pullback.log x
    sqrt (TensorElement x) = TensorElement $ unaryVJP sqrt Pullback.sqrt x
    sin (TensorElement x) = TensorElement $ unaryVJP sin Pullback.sin x
    cos (TensorElement x) = TensorElement $ unaryVJP cos Pullback.cos x
    tan (TensorElement x) = TensorElement $ unaryVJP tan Pullback.tan x
    asin (TensorElement x) = TensorElement $ unaryVJP asin Pullback.asin x
    acos (TensorElement x) = TensorElement $ unaryVJP acos Pullback.acos x
    atan (TensorElement x) = TensorElement $ unaryVJP atan Pullback.atan x
    sinh (TensorElement x) = TensorElement $ unaryVJP sinh Pullback.sinh x
    cosh (TensorElement x) = TensorElement $ unaryVJP cosh Pullback.cosh x
    tanh (TensorElement x) = TensorElement $ unaryVJP tanh Pullback.tanh x
    asinh (TensorElement x) = TensorElement $ unaryVJP asinh Pullback.asinh x
    acosh (TensorElement x) = TensorElement $ unaryVJP acosh Pullback.acosh x
    atanh (TensorElement x) = TensorElement $ unaryVJP atanh Pullback.atanh x
    (**) (TensorElement x) (TensorElement y) = TensorElement $ binaryVJP (**) Pullback.power x y
    logBase x y = log y / log x
