{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Integral division, rounding, and floating-point properties.
module Synapse.Tensor.Ops.Numeric
    ( Integral (..)
    , RealFrac (..)
    , RealFloat (..)
    ) where

import Data.Int (Int64)

import Synapse.Tensor.Ops.Logic (Eq (..), Ord (..))
import Synapse.Tensor.Ops.Raw.Elementwise (map, zipWith)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Prelude hiding
    ( Eq
    , Integral
    , Ord
    , RealFloat
    , RealFrac
    , atan2
    , ceiling
    , compare
    , decodeFloat
    , div
    , divMod
    , encodeFloat
    , exponent
    , floatDigits
    , floatRadix
    , floatRange
    , floor
    , isDenormalized
    , isIEEE
    , isInfinite
    , isNaN
    , isNegativeZero
    , map
    , max
    , min
    , mod
    , properFraction
    , quot
    , quotRem
    , rem
    , round
    , scaleFloat
    , significand
    , truncate
    , zipWith
    , (/=)
    , (<)
    , (<=)
    , (==)
    , (>)
    , (>=)
    )

import qualified Data.Array.Accelerate as A
import qualified Prelude as P

-- | Integer division; divisors must be nonzero and the quotient must be representable.
class (Num a, Ord a) => Integral a where
    -- | Divide corresponding values, rounding toward zero.
    quot
        :: a
        -> a
        -> a

    -- | Remainder whose sign follows the dividend.
    rem
        :: a
        -> a
        -> a

    -- | Divide corresponding values, rounding toward negative infinity.
    div
        :: a
        -> a
        -> a

    -- | Remainder whose sign follows the divisor.
    mod
        :: a
        -> a
        -> a

    -- | Quotient toward zero and its corresponding remainder.
    quotRem
        :: a
        -> a
        -> (a, a)
    quotRem x y = (quot x y, rem x y)

    -- | Quotient toward negative infinity and its corresponding remainder.
    divMod
        :: a
        -> a
        -> (a, a)
    divMod x y = (div x y, mod x y)

instance (A.Integral a, Ord (TensorElement a), P.Ord a) => Integral (TensorElement a) where
    quot (TensorElement x) (TensorElement y) = TensorElement (A.quot x y)
    rem (TensorElement x) (TensorElement y) = TensorElement (A.rem x y)
    div (TensorElement x) (TensorElement y) = TensorElement (A.div x y)
    mod (TensorElement x) (TensorElement y) = TensorElement (A.mod x y)
instance (Shape sh, DType a, Integral (TensorElement a), Num (Tensor sh a), P.Integral a) => Integral (Tensor sh a) where
    quot = zipWith quot
    rem = zipWith rem
    div = zipWith div
    mod = zipWith mod

-- | Fractional decomposition and rounding; inputs must be finite and integer results must fit in Int64.
class (Ord a, Fractional a) => RealFrac a where
    -- | Int64 element expressions or tensors with the input shape.
    type IntegralResult a

    -- | Split each value into its integral part toward zero and fractional remainder.
    properFraction
        :: a
        -> (IntegralResult a, a)

    -- | Round toward zero.
    truncate
        :: a
        -> IntegralResult a

    -- | Round to the nearest integer, choosing the even integer at ties.
    round
        :: a
        -> IntegralResult a

    -- | Round toward positive infinity.
    ceiling
        :: a
        -> IntegralResult a

    -- | Round toward negative infinity.
    floor
        :: a
        -> IntegralResult a

-- | Map a paired element result to two tensors of the original shape.
mapPair
    :: (Shape sh, DType a, DType b, DType c)
    => (TensorElement a -> (TensorElement b, TensorElement c))
    -> Tensor sh a
    -> (Tensor sh b, Tensor sh c)
mapPair f (Tensor input) =
    let (left, right) = A.unzip $ A.map pair input
        pair x = let (TensorElement y, TensorElement z) = f (TensorElement x) in A.lift (y, z)
     in (Tensor left, Tensor right)

instance (A.RealFrac a, Ord (TensorElement a), P.Ord a) => RealFrac (TensorElement a) where
    type IntegralResult (TensorElement a) = TensorElement Int64
    properFraction (TensorElement x) =
        let small = abs x A.< 1
            (n, r) = A.properFraction (A.cond small 1 x)
         in (TensorElement $ A.cond small 0 n, TensorElement $ A.cond small x r)
    truncate (TensorElement x) = TensorElement (A.truncate x)
    round (TensorElement x) =
        let (TensorElement n, TensorElement fraction) = properFraction (TensorElement x)
            distance = abs fraction
            away = A.cond (fraction A.< 0) (n - 1) (n + 1)
            roundAway = distance A.> 0.5 A.|| (distance A.== 0.5 A.&& A.odd n)
         in TensorElement $ A.cond roundAway away n
    ceiling (TensorElement x) = TensorElement (A.ceiling x)
    floor (TensorElement x) = TensorElement (A.floor x)
instance
    ( Shape sh
    , DType a
    , RealFrac (TensorElement a)
    , IntegralResult (TensorElement a) ~ TensorElement Int64
    , Fractional (Tensor sh a)
    , P.RealFrac a
    )
    => RealFrac (Tensor sh a)
    where
    type IntegralResult (Tensor sh a) = Tensor sh Int64
    properFraction = mapPair properFraction
    truncate = map truncate
    round = map round
    ceiling = map ceiling
    floor = map floor

-- | Floating-point decomposition, scaling, classification, and two-argument arctangent.
class (RealFrac a, Floating a) => RealFloat a where
    -- | Int element expressions or tensors with the input shape.
    type ExponentResult a

    -- | Radix of the floating-point representation.
    floatRadix
        :: a
        -> IntegralResult a

    -- | Number of radix digits in the significand.
    floatDigits
        :: a
        -> ExponentResult a

    -- | Minimum and maximum exponents of normalized values.
    floatRange
        :: a
        -> (ExponentResult a, ExponentResult a)

    -- | Decode a finite value into an integer significand and exponent.
    decodeFloat
        :: a
        -> (IntegralResult a, ExponentResult a)

    -- | Construct a value as @significand * radix^exponent@.
    encodeFloat
        :: IntegralResult a
        -> ExponentResult a
        -> a

    -- | Exponent associated with the normalized significand of a finite value.
    exponent
        :: a
        -> ExponentResult a

    -- | Normalized fractional significand of a finite value.
    significand
        :: a
        -> a

    -- | Multiply each value by an integral power of its radix.
    scaleFloat
        :: ExponentResult a
        -> a
        -> a

    -- | Test whether each value is NaN.
    isNaN
        :: a
        -> Comparison a

    -- | Test whether each value is positive or negative infinity.
    isInfinite
        :: a
        -> Comparison a

    -- | Test whether each value is subnormal.
    isDenormalized
        :: a
        -> Comparison a

    -- | Test whether each value is negative zero.
    isNegativeZero
        :: a
        -> Comparison a

    -- | Test whether the representation follows IEEE floating-point arithmetic.
    isIEEE
        :: a
        -> Comparison a

    -- | Angle of the point @(x, y)@; arguments are @y@ followed by @x@.
    atan2
        :: a
        -> a
        -> a

instance (A.RealFloat a, Ord (TensorElement a), P.Ord a) => RealFloat (TensorElement a) where
    type ExponentResult (TensorElement a) = TensorElement Int
    floatRadix (TensorElement x) = TensorElement (A.floatRadix x)
    floatDigits (TensorElement x) = TensorElement (A.floatDigits x)
    floatRange (TensorElement x) = let (lo, hi) = A.floatRange x in (TensorElement lo, TensorElement hi)
    decodeFloat (TensorElement x) = let (m, e) = A.decodeFloat x in (TensorElement m, TensorElement e)
    encodeFloat (TensorElement m) (TensorElement e) = TensorElement (A.encodeFloat m e)
    exponent (TensorElement x) = TensorElement (A.exponent x)
    significand (TensorElement x) = TensorElement (A.significand x)
    scaleFloat (TensorElement e) (TensorElement x) = TensorElement (A.scaleFloat e x)
    isNaN (TensorElement x) = TensorElement (A.isNaN x)
    isInfinite (TensorElement x) = TensorElement (A.isInfinite x)
    isDenormalized (TensorElement x) =
        let magnitude = abs x
            smallestNormal = A.encodeFloat 1 (fst (A.floatRange x) - 1)
         in TensorElement $ magnitude A.> 0 A.&& magnitude A.< smallestNormal
    isNegativeZero (TensorElement x) = TensorElement (A.isNegativeZero x)
    isIEEE (TensorElement x) = TensorElement (A.isIEEE x)
    atan2 (TensorElement y) (TensorElement x) = TensorElement (A.atan2 y x)
instance
    ( Shape sh
    , DType a
    , RealFloat (TensorElement a)
    , IntegralResult (TensorElement a) ~ TensorElement Int64
    , ExponentResult (TensorElement a) ~ TensorElement Int
    , Floating (Tensor sh a)
    , P.RealFloat a
    )
    => RealFloat (Tensor sh a)
    where
    type ExponentResult (Tensor sh a) = Tensor sh Int
    floatRadix = map floatRadix
    floatDigits = map floatDigits
    floatRange = mapPair floatRange
    decodeFloat = mapPair decodeFloat
    encodeFloat = zipWith encodeFloat
    exponent = map exponent
    significand = map significand
    scaleFloat = zipWith scaleFloat
    isNaN = map isNaN
    isInfinite = map isInfinite
    isDenormalized = map isDenormalized
    isNegativeZero = map isNegativeZero
    isIEEE = map isIEEE
    atan2 = zipWith atan2
