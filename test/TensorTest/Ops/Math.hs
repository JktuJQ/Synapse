{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Math (tests) where

import Control.Monad (forM_)
import Data.Int (Int32, Int64)
import Data.Word (Word32, Word64)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import qualified TensorTest.Ops.Math.Linear
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "math"
    [ TensorTest.Ops.Math.Linear.tests,
      testProperty "integer arithmetic agrees with lists" propIntegerArithmetic,
      testProperty "Floating methods agree with host calculations" (H.withTests 1 propFloatingMethods),
      testProperty "Num instance is elementwise" (H.withTests 1 propNumInstance),
      testProperty "Fractional instance is elementwise" (H.withTests 1 propFractionalInstance),
      testProperty "Floating instance is elementwise" (H.withTests 1 propFloatingInstance),
      testProperty "tensor comparisons and extrema agree with lists" propComparisons,
      testProperty "element comparisons compose inside map" (H.withTests 1 propElementComparisons),
      testProperty "floating comparisons follow IEEE semantics" (H.withTests 1 propFloatingComparisons),
      testProperty "comparisons support scalars and empty tensors" (H.withTests 1 propComparisonEdges),
      testProperty "comparisons support all scalar dtypes" (H.withTests 1 propComparisonDTypes),
      testProperty "floating extrema preserve infinities" (H.withTests 1 propInfiniteExtrema),
      testProperty "valid comparison types compile" (H.withTests 1 (Compiler.accepts "test/TensorTest/Ops/Math/Fixtures/Accepted/Valid.hs")),
      testProperty "comparison shapes must match" (H.withTests 1 (Compiler.rejects "test/TensorTest/Ops/Math/Fixtures/Rejected/ComparisonShape.hs" "Couldn't match type"))
    ]

propNumInstance :: H.Property
propNumInstance = H.property $ do
  let lhs = T.fromList @'[2, 2] [1, -2, 3, -4] :: T.Tensor '[2, 2] Int
      rhs = T.fromList @'[2, 2] [10, 20, 30, 40] :: T.Tensor '[2, 2] Int
      literal = 3 :: T.Tensor '[2, 2] Int
  T.toList (T.run T.CPU (lhs + rhs)) H.=== [11, 18, 33, 36]
  T.toList (T.run T.CPU (rhs - lhs)) H.=== [9, 22, 27, 44]
  T.toList (T.run T.CPU (lhs * rhs)) H.=== [10, -40, 90, -160]
  T.toList (T.run T.CPU (negate lhs)) H.=== [-1, 2, -3, 4]
  T.toList (T.run T.CPU (abs lhs)) H.=== [1, 2, 3, 4]
  T.toList (T.run T.CPU (signum lhs)) H.=== [1, -1, 1, -1]
  T.toList (T.run T.CPU literal) H.=== [3, 3, 3, 3]

propFractionalInstance :: H.Property
propFractionalInstance = H.property $ do
  let lhs = T.fromList @'[2] [8.0, 9.0] :: T.Tensor '[2] T.Float32
      rhs = T.fromList @'[2] [2.0, 3.0] :: T.Tensor '[2] T.Float32
      literal = 0.5 :: T.Tensor '[2] T.Float32
  T.toList (T.run T.CPU (lhs / rhs)) H.=== [4.0, 3.0]
  T.toList (T.run T.CPU (recip rhs)) H.=== [0.5, 1.0 / 3.0]
  T.toList (T.run T.CPU literal) H.=== [0.5, 0.5]

propFloatingInstance :: H.Property
propFloatingInstance = H.property $ do
  let tensor = T.fromList @'[2] [1.0, 4.0] :: T.Tensor '[2] T.Float32
      expInput = T.fromList @'[2] [0.0, 1.0] :: T.Tensor '[2] T.Float32
      logInput = T.fromList @'[2] [1.0, Prelude.exp 1.0] :: T.Tensor '[2] T.Float32
      piTensor = pi :: T.Tensor '[2] T.Float32
  approxList (T.toList (T.run T.CPU (sqrt tensor))) [1.0, 2.0]
  approxList (T.toList (T.run T.CPU (exp expInput))) [1.0, Prelude.exp 1.0]
  approxList (T.toList (T.run T.CPU (log logInput))) [0.0, 1.0]
  approxList (T.toList (T.run T.CPU piTensor)) [Prelude.pi, Prelude.pi]

propIntegerArithmetic :: H.Property
propIntegerArithmetic = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let lhs = T.fromList @'[2, 3] xs
      rhs = T.fromList @'[2, 3] ys
  T.toList (T.run T.CPU (lhs + rhs)) H.=== zipWith (+) xs ys
  T.toList (T.run T.CPU (lhs - rhs)) H.=== zipWith (-) xs ys
  T.toList (T.run T.CPU (lhs * rhs)) H.=== zipWith (*) xs ys
  T.toList (T.run T.CPU (abs lhs)) H.=== map abs xs
  T.toList (T.run T.CPU (signum lhs)) H.=== map signum xs
  T.toList (T.run T.CPU (negate lhs)) H.=== map negate xs

propFloatingMethods :: H.Property
propFloatingMethods = H.property $ do
  let xs = [0.2, 0.3, 0.5, 0.7 :: T.Float64]
      input = T.fromList @'[4] xs
      operations :: [(String, T.Tensor '[4] T.Float64 -> T.Tensor '[4] T.Float64, T.Float64 -> T.Float64)]
      operations =
        [ ("exp", exp, exp),
          ("log", log, log),
          ("sqrt", sqrt, sqrt),
          ("sin", sin, sin),
          ("cos", cos, cos),
          ("tan", tan, tan),
          ("asin", asin, asin),
          ("acos", acos, acos),
          ("atan", atan, atan),
          ("sinh", sinh, sinh),
          ("cosh", cosh, cosh),
          ("tanh", tanh, tanh),
          ("asinh", asinh, asinh),
          ("acosh", acosh . (+ 1), acosh . (+ 1)),
          ("atanh", atanh, atanh),
          ("power", (** 2.5), (** 2.5)),
          ("logBase", logBase 2, logBase 2)
        ]
  forM_ operations $ \(name, tensorOp, hostOp) -> do
    H.annotate name
    approxList (T.toList (T.run T.CPU (tensorOp input))) (map hostOp xs)

propComparisons :: H.Property
propComparisons = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let lhs = T.fromList @'[2, 3] xs
      rhs = T.fromList @'[2, 3] ys
  T.toList (T.run T.CPU (lhs T.== rhs)) H.=== zipWith (==) xs ys
  T.toList (T.run T.CPU (lhs T./= rhs)) H.=== zipWith (/=) xs ys
  T.toList (T.run T.CPU (lhs T.< rhs)) H.=== zipWith (<) xs ys
  T.toList (T.run T.CPU (lhs T.<= rhs)) H.=== zipWith (<=) xs ys
  T.toList (T.run T.CPU (lhs T.> rhs)) H.=== zipWith (>) xs ys
  T.toList (T.run T.CPU (lhs T.>= rhs)) H.=== zipWith (>=) xs ys
  T.toList (T.run T.CPU (T.min lhs rhs)) H.=== zipWith min xs ys
  T.toList (T.run T.CPU (T.max lhs rhs)) H.=== zipWith max xs ys

propElementComparisons :: H.Property
propElementComparisons = H.property $ do
  let input = T.fromList @'[3] [-1, 0, 1 :: Int]
  T.toList (T.run T.CPU (input T.> 0)) H.=== [False, False, True]
  T.toList (T.run T.CPU (input T.== 0)) H.=== [False, True, False]
  T.toList (T.run T.CPU (T.max input 0)) H.=== [0, 0, 1]
  T.toList (T.run T.CPU (T.sum input T.== 0)) H.=== [True]

propFloatingComparisons :: H.Property
propFloatingComparisons = H.property $ do
  let lhs = T.fromList @'[3] [0 / 0, 2, 3 :: Double]
      rhs = T.fromList @'[3] [1, 2, 0 / 0 :: Double]
  T.toList (T.run T.CPU (lhs T.== rhs)) H.=== [False, True, False]
  T.toList (T.run T.CPU (lhs T./= rhs)) H.=== [True, False, True]
  T.toList (T.run T.CPU (lhs T.< rhs)) H.=== [False, False, False]
  T.toList (T.run T.CPU (lhs T.<= rhs)) H.=== [False, True, False]
  T.toList (T.run T.CPU (lhs T.> rhs)) H.=== [False, False, False]
  T.toList (T.run T.CPU (lhs T.>= rhs)) H.=== [False, True, False]
  let minima = T.toList (T.run T.CPU (T.min lhs rhs))
      maxima = T.toList (T.run T.CPU (T.max lhs rhs))
  case (minima, maxima) of
    ([lowFirst, lowMiddle, lowLast], [highFirst, highMiddle, highLast]) -> do
      H.assert (isNaN lowFirst && isNaN lowLast)
      H.assert (isNaN highFirst && isNaN highLast)
      lowMiddle H.=== 2
      highMiddle H.=== 2
    _ -> H.failure

propComparisonEdges :: H.Property
propComparisonEdges = H.property $ do
  let scalar = T.scalar (T.constant True)
      empty = T.zeros @'[2, 0] :: T.Tensor '[2, 0] Int
  T.toList (T.run T.CPU (scalar T.== scalar)) H.=== [True]
  let lhs = T.fromList @'[4] [False, False, True, True]
      rhs = T.fromList @'[4] [False, True, False, True]
  T.toList (T.run T.CPU (lhs T.< rhs)) H.=== [False, True, False, False]
  T.toList (T.run T.CPU (lhs T.<= rhs)) H.=== [True, True, False, True]
  T.toList (T.run T.CPU (lhs T.> rhs)) H.=== [False, False, True, False]
  T.toList (T.run T.CPU (lhs T.>= rhs)) H.=== [True, False, True, True]
  T.toList (T.run T.CPU (T.min lhs rhs)) H.=== [False, False, False, True]
  T.toList (T.run T.CPU (T.max lhs rhs)) H.=== [False, True, True, True]
  T.toList (T.run T.CPU (empty T.< empty)) H.=== []
  T.toList (T.run T.CPU (T.max empty empty)) H.=== []

propComparisonDTypes :: H.Property
propComparisonDTypes = H.property $ do
  let check :: (T.DType a, T.Ord (T.TensorElement a)) => a -> a -> H.PropertyT IO ()
      check low high = do
        let lhs = T.scalar (T.constant low)
            rhs = T.scalar (T.constant high)
        T.toList (T.run T.CPU (lhs T.< rhs)) H.=== [True]
        T.toList (T.run T.CPU (lhs T.== lhs)) H.=== [True]
        T.toList (T.run T.CPU (lhs T./= rhs)) H.=== [True]
  check (-1 :: Int) 2
  check (-1 :: Int32) 2
  check (-1 :: Int64) 2
  check (0 :: Word32) 2
  check (0 :: Word64) 2
  check False True
  check (-1 :: Float) 2
  check (-1 :: Double) 2
  let lhs = T.fromList @'[2] [(1 :: Int, True), (2, False)]
      rhs = T.fromList @'[2] [(1 :: Int, True), (2, True)]
  T.toList (T.run T.CPU (lhs T.== rhs)) H.=== [True, False]
  T.toList (T.run T.CPU (lhs T./= rhs)) H.=== [False, True]
  let nan = T.scalar (T.constant (0 / 0 :: Float))
  T.toList (T.run T.CPU (nan T.== nan)) H.=== [False]
  T.toList (T.run T.CPU (nan T./= nan)) H.=== [True]
  H.assert (all isNaN (T.toList (T.run T.CPU (T.min nan nan))))
  H.assert (all isNaN (T.toList (T.run T.CPU (T.max nan nan))))

propInfiniteExtrema :: H.Property
propInfiniteExtrema = H.property $ do
  let lhs = T.fromList @'[3] [-(1 / 0), 1 / 0, 2 :: Double]
      rhs = T.fromList @'[3] [3, 4, -(1 / 0) :: Double]
  T.toList (T.run T.CPU (T.min lhs rhs)) H.=== [-(1 / 0), 4, -(1 / 0)]
  T.toList (T.run T.CPU (T.max lhs rhs)) H.=== [3, 1 / 0, 2]
  let lhsFloat = T.fromList @'[2] [-(1 / 0), 1 / 0 :: Float]
      rhsFloat = T.zeros @'[2]
  T.toList (T.run T.CPU (T.min lhsFloat rhsFloat)) H.=== [-(1 / 0), 0]
  T.toList (T.run T.CPU (T.max lhsFloat rhsFloat)) H.=== [0, 1 / 0]
