{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

{- | Rounding and floating-point operations for differentiable values.
Integer results and representation metadata are detached from the differentiation graph.
The fractional remainder has derivative one; significand differentiation holds the exponent fixed.
-}
module Synapse.Autograd.Tensor.Ops.Numeric () where

import Data.Int (Int64)
import qualified Synapse.Tensor.Representation as T

import Synapse.Autograd.Representation (Context)
import Synapse.Autograd.Tensor.Ops (binaryVJP, unaryVJP)
import Synapse.Autograd.Tensor.Ops.Logic ()
import Synapse.Autograd.Tensor.Ops.Math ()
import Synapse.Autograd.Tensor.Representation (Tensor (..), TensorElement (..))
import Synapse.Tensor.Ops.Constraints (RealFloatDType, RealFracDType)
import Synapse.Tensor.Shapes.Dimensions (Shape)

import qualified Numeric.Backprop.Explicit as BPE
import qualified Numeric.Backprop.Internal as BPI

import qualified Synapse.Autograd.Tensor.Ops.Pullback.Numeric as Pullback
import qualified Synapse.Tensor.Ops.Numeric as E

instance
    (Context s, Shape sh, RealFracDType a)
    => E.RealFrac (Tensor s sh a)
    where
    type IntegralResult (Tensor s sh a) = T.Tensor sh Int64
    properFraction (Tensor x) =
        let (integral, _) = E.properFraction (BPI._bvVal x)
            fraction = Tensor $ unaryVJP (snd . E.properFraction) (\_ _ seed -> seed) x
         in (integral, fraction)
    truncate (Tensor x) = E.truncate (BPI._bvVal x)
    round (Tensor x) = E.round (BPI._bvVal x)
    ceiling (Tensor x) = E.ceiling (BPI._bvVal x)
    floor (Tensor x) = E.floor (BPI._bvVal x)
instance (Context s, RealFracDType a) => E.RealFrac (TensorElement s a) where
    type IntegralResult (TensorElement s a) = T.TensorElement Int64
    properFraction (TensorElement x) =
        let (integral, _) = E.properFraction (BPI._bvVal x)
            fraction = TensorElement $ unaryVJP (snd . E.properFraction) (\_ _ seed -> seed) x
         in (integral, fraction)
    truncate (TensorElement x) = E.truncate (BPI._bvVal x)
    round (TensorElement x) = E.round (BPI._bvVal x)
    ceiling (TensorElement x) = E.ceiling (BPI._bvVal x)
    floor (TensorElement x) = E.floor (BPI._bvVal x)
instance
    (Context s, Shape sh, RealFloatDType a)
    => E.RealFloat (Tensor s sh a)
    where
    type ExponentResult (Tensor s sh a) = T.Tensor sh Int
    floatRadix (Tensor x) = E.floatRadix (BPI._bvVal x)
    floatDigits (Tensor x) = E.floatDigits (BPI._bvVal x)
    floatRange (Tensor x) = E.floatRange (BPI._bvVal x)
    decodeFloat (Tensor x) = E.decodeFloat (BPI._bvVal x)
    encodeFloat m e = Tensor $ BPE.auto (E.encodeFloat m e)
    exponent (Tensor x) = E.exponent (BPI._bvVal x)
    significand (Tensor x) = Tensor $ unaryVJP E.significand Pullback.significand x
    scaleFloat power (Tensor x) = Tensor $ unaryVJP (E.scaleFloat power) (Pullback.scaleFloat power) x
    isNaN (Tensor x) = E.isNaN (BPI._bvVal x)
    isInfinite (Tensor x) = E.isInfinite (BPI._bvVal x)
    isDenormalized (Tensor x) = E.isDenormalized (BPI._bvVal x)
    isNegativeZero (Tensor x) = E.isNegativeZero (BPI._bvVal x)
    isIEEE (Tensor x) = E.isIEEE (BPI._bvVal x)
    atan2 (Tensor y) (Tensor x) = Tensor $ binaryVJP E.atan2 Pullback.atan2 y x
instance (Context s, RealFloatDType a) => E.RealFloat (TensorElement s a) where
    type ExponentResult (TensorElement s a) = T.TensorElement Int
    floatRadix (TensorElement x) = E.floatRadix (BPI._bvVal x)
    floatDigits (TensorElement x) = E.floatDigits (BPI._bvVal x)
    floatRange (TensorElement x) = E.floatRange (BPI._bvVal x)
    decodeFloat (TensorElement x) = E.decodeFloat (BPI._bvVal x)
    encodeFloat m e = TensorElement $ BPE.auto (E.encodeFloat m e)
    exponent (TensorElement x) = E.exponent (BPI._bvVal x)
    significand (TensorElement x) = TensorElement $ unaryVJP E.significand Pullback.significand x
    scaleFloat power (TensorElement x) =
        TensorElement $ unaryVJP (E.scaleFloat power) (Pullback.scaleFloat power) x
    isNaN (TensorElement x) = E.isNaN (BPI._bvVal x)
    isInfinite (TensorElement x) = E.isInfinite (BPI._bvVal x)
    isDenormalized (TensorElement x) = E.isDenormalized (BPI._bvVal x)
    isNegativeZero (TensorElement x) = E.isNegativeZero (BPI._bvVal x)
    isIEEE (TensorElement x) = E.isIEEE (BPI._bvVal x)
    atan2 (TensorElement y) (TensorElement x) = TensorElement $ binaryVJP E.atan2 Pullback.atan2 y x
