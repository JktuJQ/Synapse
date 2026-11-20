module TensorTest.Ops.Elementwise (tests) where

import Data.Int (Int64)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
    testGroup
        "elementwise"
        [ testProperty "boolean operators agree with truth tables" (H.withTests 1 propBoolean)
        , testProperty "tensor masks select corresponding values" propSelect
        , testProperty "element conditions compose inside generators" (H.withTests 1 propElementSelect)
        , testProperty "scalar conditions select whole tensor computations" (H.withTests 1 propCond)
        , testProperty "selection supports scalars, empty tensors, and tuples" (H.withTests 1 propSelectEdges)
        , testProperty "integer quotients truncate toward zero" propIntegral
        , testProperty "floating classification distinguishes NaN and infinities" (H.withTests 1 propClassification)
        , testProperty "comparison returns staged ordering values" (H.withTests 1 propCompare)
        , testProperty "rounding follows Prelude and preserves shapes" (H.withTests 1 propRealFrac)
        , testProperty "floating decomposition and scaling follow Prelude" (H.withTests 1 propRealFloat)
        , testProperty "IEEE predicates distinguish signed zero and subnormals" (H.withTests 1 propIEEE)
        , testProperty "numeric conversions preserve source shape" (H.withTests 1 propToFloating)
        , testProperty
            "valid conditional operations compile"
            (H.withTests 1 (Compiler.accepts (root ++ "Accepted/Control.hs")))
        , testProperty
            "select mask shape must match"
            (H.withTests 1 (Compiler.rejects (root ++ "Rejected/SelectMask.hs") "Couldn't match type"))
        , testProperty
            "select branch shapes must match"
            (H.withTests 1 (Compiler.rejects (root ++ "Rejected/SelectShape.hs") "Couldn't match type"))
        , testProperty
            "cond branch shapes must match"
            (H.withTests 1 (Compiler.rejects (root ++ "Rejected/CondShape.hs") "Couldn't match type"))
        , testProperty
            "boolean operand shapes must match"
            (H.withTests 1 (Compiler.rejects (root ++ "Rejected/BooleanShape.hs") "Couldn't match type"))
        ]
  where
    root = "test/TensorTest/Ops/Elementwise/Fixtures/"

propBoolean :: H.Property
propBoolean = H.property $ do
    let xs = [False, False, True, True]
        ys = [False, True, False, True]
        lhs = T.fromList @'[4] xs
        rhs = T.fromList @'[4] ys
    T.toList (T.run T.CPU (T.not lhs)) H.=== map not xs
    T.toList (T.run T.CPU (lhs T.&& rhs)) H.=== zipWith (&&) xs ys
    T.toList (T.run T.CPU (lhs T.|| rhs)) H.=== zipWith (||) xs ys

propSelect :: H.Property
propSelect = H.property $ do
    xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
    ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
    flags <- H.forAll $ Gen.list (Range.singleton 6) Gen.bool
    let lhs = T.fromList @'[2, 3] xs
        rhs = T.fromList @'[2, 3] ys
        mask = T.fromList @'[2, 3] flags
    T.toList (T.run T.CPU (T.select mask lhs rhs)) H.=== zipWith3 (\p x y -> if p then x else y) flags xs ys
    T.toList (T.run T.CPU (T.select ((lhs T.> 0) T.&& (lhs T.< 50)) lhs 0))
        H.=== map (\x -> if x > 0 && x < 50 then x else 0) xs

propElementSelect :: H.Property
propElementSelect = H.property $ do
    let input = T.fromList @'[3] [-2, 0, 3 :: Int]
    T.toList (T.run T.CPU (T.fromGen @'[3] (\ix -> let x = input T.!! ix in T.select (x T.< 0) (negate x) x)))
        H.=== [2, 0, 3]
    T.toList (T.run T.CPU (T.fromGen @'[3] (\ix -> let x = input T.!! ix in T.cond (x T.> 0) (x * 2) x)))
        H.=== [-2, 0, 6]

propCond :: H.Property
propCond = H.property $ do
    let lhs = T.fromList @'[3] [1, 2, 3 :: Int]
        rhs = T.fromList @'[3] [4, 5, 6 :: Int]
    T.toList (T.run T.CPU (T.cond (T.sum lhs T.> 0) (lhs + 1) rhs)) H.=== [2, 3, 4]
    T.toList (T.run T.CPU (T.cond (T.scalar (T.element False)) lhs rhs)) H.=== [4, 5, 6]

propSelectEdges :: H.Property
propSelectEdges = H.property $ do
    let empty = T.zeros @'[2, 0] :: T.Tensor '[2, 0] Int
        emptyMask = T.full @'[2, 0] True
        scalarMask = T.scalar (T.element False)
    T.toList (T.run T.CPU (T.select emptyMask empty empty)) H.=== []
    T.toList (T.run T.CPU (T.not emptyMask T.|| emptyMask)) H.=== []
    T.toList (T.run T.CPU (T.select scalarMask (T.scalar (T.element True)) (T.scalar (T.element False))))
        H.=== [False]
    let lhs = T.fromList @'[2] [(1 :: Int, True), (2, False)]
        rhs = T.fromList @'[2] [(3 :: Int, False), (4, True)]
    T.toList (T.run T.CPU (T.select (T.fromList @'[2] [True, False]) lhs rhs)) H.=== [(1, True), (4, True)]

propIntegral :: H.Property
propIntegral = H.property $ do
    xs <- H.forAll $ Gen.list (Range.singleton 4) (Gen.int (Range.linear (-100) 100))
    ys <- H.forAll $ Gen.list (Range.singleton 4) (Gen.element [-7, -3, 2, 5])
    let lhs = T.fromList @'[4] xs
        rhs = T.fromList @'[4] ys
    T.toList (T.run T.CPU (T.quot lhs rhs)) H.=== zipWith quot xs ys
    T.toList (T.run T.CPU (T.rem lhs rhs)) H.=== zipWith rem xs ys
    T.toList (T.run T.CPU (T.quot lhs 3)) H.=== map (`quot` 3) xs
    T.toList (T.run T.CPU (T.rem lhs (-3))) H.=== map (`rem` (-3)) xs
    T.toList (T.run T.CPU (T.div lhs rhs)) H.=== zipWith div xs ys
    T.toList (T.run T.CPU (T.mod lhs rhs)) H.=== zipWith mod xs ys
    let (quotients, remainders) = T.quotRem lhs rhs
        (divisors, moduli) = T.divMod lhs rhs
    T.toList (T.run T.CPU quotients) H.=== zipWith quot xs ys
    T.toList (T.run T.CPU remainders) H.=== zipWith rem xs ys
    T.toList (T.run T.CPU divisors) H.=== zipWith div xs ys
    T.toList (T.run T.CPU moduli) H.=== zipWith mod xs ys

propClassification :: H.Property
propClassification = H.property $ do
    let input = T.fromList @'[5] [0 / 0, 1 / 0, -(1 / 0), 0, -3 :: Double] :: T.Tensor '[5] Double
        single = T.fromList @'[3] [0 / 0, 1 / 0, 2 :: Float] :: T.Tensor '[3] Float
    T.toList (T.run T.CPU (T.isNaN input)) H.=== [True, False, False, False, False]
    T.toList (T.run T.CPU (T.isInfinite input)) H.=== [False, True, True, False, False]
    T.toList (T.run T.CPU (T.isNaN single)) H.=== [True, False, False]
    T.toList (T.run T.CPU (T.isInfinite single)) H.=== [False, True, False]
    T.toList (T.run T.CPU (T.isNaN (T.zeros @'[0] :: T.Tensor '[0] Double))) H.=== []

propCompare :: H.Property
propCompare = H.property $ do
    let xs = [1, 3, 2 :: Int]
        ys = [2, 1, 2 :: Int]
        lhs = T.fromList @'[3] xs :: T.Tensor '[3] Int
        rhs = T.fromList @'[3] ys :: T.Tensor '[3] Int
    T.toList (T.run T.CPU (T.compare lhs rhs)) H.=== zipWith compare xs ys
    let bools = T.fromList @'[2] [False, True] :: T.Tensor '[2] Bool
    T.toList (T.run T.CPU (T.compare bools (T.full True))) H.=== [LT, EQ]
    T.toList (T.run T.CPU (T.fromGen @'[3] (\ix -> T.compare (lhs T.!! ix) (rhs T.!! ix))))
        H.=== zipWith compare xs ys

propRealFrac :: H.Property
propRealFrac = H.property $ do
    let xs = [-2.75, -2.5, -1.5, -0.25, 0.25, 0.5, 1.5, 2.5 :: Double]
        input = T.fromList @'[2, 4] xs
        (whole, fractional) = T.properFraction input
    T.toList (T.run T.CPU whole) H.=== map truncate xs
    T.toList (T.run T.CPU fractional) H.=== map (snd . (properFraction :: Double -> (Int64, Double))) xs
    T.toList (T.run T.CPU (T.truncate input)) H.=== map truncate xs
    T.toList (T.run T.CPU (T.round input)) H.=== map round xs
    T.toList (T.run T.CPU (T.ceiling input)) H.=== map ceiling xs
    T.toList (T.run T.CPU (T.floor input)) H.=== map floor xs
    let single = T.fromList @'[4] [-1.5, 0.5, 1.5, 2.5] :: T.Tensor '[4] Float
    T.toList (T.run T.CPU (T.round single)) H.=== [-2, 0, 2, 2]
    T.toList (T.run T.CPU (T.floor (T.fromList @'[0] [] :: T.Tensor '[0] Double))) H.=== ([] :: [Int64])
    T.toList (T.run T.CPU (T.round (T.full @'[] 1.5 :: T.Tensor '[] Double))) H.=== [2]
    let tiny = [encodeFloat 1 (-1074), -1e-30, 0, 1e-30 :: Double]
        (tinyWhole, tinyFraction) = T.properFraction (T.fromList @'[4] tiny :: T.Tensor '[4] Double)
    T.toList (T.run T.CPU tinyWhole) H.=== [0, 0, 0, 0]
    T.toList (T.run T.CPU tinyFraction) H.=== tiny

propRealFloat :: H.Property
propRealFloat = H.property $ do
    let xs = [-6, -0.375, 0, 0.375, 1.5, 9 :: Double]
        input = T.fromList @'[6] xs
        powers = T.fromList @'[6] [-3, -2, -1, 0, 1, 2 :: Int]
        (mantissas, exponents) = T.decodeFloat input
        (lo, hi) = T.floatRange input
    T.toList (T.run T.CPU mantissas) H.=== map (fromInteger . fst . decodeFloat) xs
    T.toList (T.run T.CPU exponents) H.=== map (snd . decodeFloat) xs
    T.toList (T.run T.CPU (T.encodeFloat mantissas exponents :: T.Tensor '[6] Double)) H.=== xs
    T.toList (T.run T.CPU (T.exponent input)) H.=== map exponent xs
    T.toList (T.run T.CPU (T.significand input)) H.=== map significand xs
    T.toList (T.run T.CPU (T.scaleFloat powers input)) H.=== zipWith scaleFloat [-3 .. 2] xs
    T.toList (T.run T.CPU (T.floatRadix input)) H.=== map (fromInteger . floatRadix) xs
    T.toList (T.run T.CPU (T.floatDigits input)) H.=== map floatDigits xs
    T.toList (T.run T.CPU lo) H.=== map (fst . floatRange) xs
    T.toList (T.run T.CPU hi) H.=== map (snd . floatRange) xs
    let single = T.fromList @'[3] [encodeFloat 1 (-149), -1.5, 2] :: T.Tensor '[3] Float
        (m, e) = T.decodeFloat single
    T.toList (T.run T.CPU (T.encodeFloat m e :: T.Tensor '[3] Float))
        H.=== [encodeFloat 1 (-149), -1.5, 2]
    T.toList (T.run T.CPU (T.atan2 input (T.full 2))) H.=== map (`atan2` 2) xs

propIEEE :: H.Property
propIEEE = H.property $ do
    let xs = [-0, 0, encodeFloat 1 (-1074), 1, 1 / 0, 0 / 0 :: Double]
        input = T.fromList @'[6] xs :: T.Tensor '[6] Double
    T.toList (T.run T.CPU (T.isNegativeZero input)) H.=== map isNegativeZero xs
    T.toList (T.run T.CPU (T.isDenormalized input)) H.=== map isDenormalized xs
    T.toList (T.run T.CPU (T.isIEEE input)) H.=== map isIEEE xs

propToFloating :: H.Property
propToFloating = H.property $ do
    let integers = T.fromList @'[3] [-3, 0, 7] :: T.Tensor '[3] Int
        singles = T.fromList @'[3] [-1.5, 0, 2.25] :: T.Tensor '[3] Float
    T.toList (T.run T.CPU (T.toFloating integers :: T.Tensor '[3] Double)) H.=== [-3, 0, 7]
    T.toList (T.run T.CPU (T.toFloating singles :: T.Tensor '[3] Double)) H.=== [-1.5, 0, 2.25]
