{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Reduction.Operations (tests) where

import Data.List (elemIndex)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh, T.DType a) => T.Tensor sh a -> [a]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "operations"
    [ testProperty "full integer reductions agree with lists" propFull,
      testProperty "axis reductions agree with row and column reductions" propAxes,
      testProperty "multiple axes retain the order of remaining axes" propMultipleAxes,
      testProperty "floating means use the number of reduced elements" (H.withTests 1 propMean),
      testProperty "empty axis lists preserve values" (H.withTests 1 propNoAxes),
      testProperty "scalar reductions preserve the scalar" (H.withTests 1 propScalar),
      testProperty "empty reduction regions yield zero or NaN" (H.withTests 1 propEmptyRegion),
      testProperty "empty retained dimensions yield empty tensors" (H.withTests 1 propEmptyOutput),
      testProperty "extrema do not use zero as an initial value" (H.withTests 1 propExtrema),
      testProperty "floating extrema propagate NaN from any position" (H.withTests 1 propNaN),
      testProperty "reductions compose with values computations" (H.withTests 1 propComposition),
      testProperty "boolean reductions agree with lists" propBoolean,
      testProperty "empty boolean regions use their identities" (H.withTests 1 propBooleanEmpty),
      testProperty "normalization is stable for large finite logits" (H.withTests 1 propNormalization),
      testProperty "normalization supports leading and middle axes" (H.withTests 1 propNormalizationAxes),
      testProperty "log sum exp supports multiple and empty axis lists" (H.withTests 1 propLogSumExpAxes),
      testProperty "log sum exp handles empty regions and special values" (H.withTests 1 propLogSumExpEdges),
      testProperty "normalization handles singleton and empty retained axes" (H.withTests 1 propNormalizationEdges),
      testProperty "arg extrema agree with first positions in lists" propArgExtrema,
      testProperty "arg extrema support leading and middle axes" (H.withTests 1 propArgAxes),
      testProperty "arg extrema use first ties and first NaN" (H.withTests 1 propArgSpecial),
      testProperty "arg extrema preserve empty outputs and singleton axes" (H.withTests 1 propArgEmpty)
    ]

propFull :: H.Property
propFull = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let input = T.fromList @'[2, 3, 4] xs
  values (T.sum input) H.=== [sum xs]
  values (T.minimum input) H.=== [minimum xs]
  values (T.maximum input) H.=== [maximum xs]

propAxes :: H.Property
propAxes = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let input = T.fromList @'[2, 3] xs
      rows = [take 3 xs, drop 3 xs]
      cols = [[xs !! j, xs !! (3 + j)] | j <- [0 .. 2]]
  values (T.sumAxes @'[1] input) H.=== fmap sum rows
  values (T.sumAxes @'[0] input) H.=== fmap sum cols
  values (T.minimumAxes @'[1] input) H.=== fmap minimum rows
  values (T.maximumAxes @'[0] input) H.=== fmap maximum cols

propMultipleAxes :: H.Property
propMultipleAxes = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let input = T.fromList @'[2, 3, 4] xs
      groups = [[xs !! ((i * 3 + j) * 4 + k) | i <- [0, 1], k <- [0 .. 3]] | j <- [0 .. 2]]
  values (T.sumAxes @'[0, 2] input) H.=== fmap sum groups
  values (T.sumAxes @'[2, 0] input) H.=== fmap sum groups
  values (T.minimumAxes @'[2, 0] input) H.=== fmap minimum groups
  values (T.maximumAxes @'[0, 2] input) H.=== fmap maximum groups
  values (T.sumAxes @'[2, 0, 1] input) H.=== [sum xs]

propMean :: H.Property
propMean = H.property $ do
  let input = T.fromList @'[2, 3] [1 .. 6 :: Double]
      higher = T.fromList @'[2, 3, 4] [0 .. 23 :: Float]
  approxList (values (T.mean input)) [3.5]
  approxList (values (T.meanAxes @'[1] input)) [2, 5]
  approxList (values (T.meanAxes @'[0] input)) [2.5, 3.5, 4.5]
  approxList (values (T.meanAxes @'[2, 0] higher)) [7.5, 11.5, 15.5]

propNoAxes :: H.Property
propNoAxes = H.property $ do
  let input = T.fromList @'[2] [-0.0, -3 :: Double]
  values (T.sumAxes @'[] input) H.=== values input
  values (T.meanAxes @'[] input) H.=== values input
  values (T.minimumAxes @'[] input) H.=== values input
  values (T.maximumAxes @'[] input) H.=== values input
  case values (T.sumAxes @'[] input) of
    first : _ -> H.assert (isNegativeZero first)
    [] -> H.failure

propScalar :: H.Property
propScalar = H.property $ do
  let input = T.scalar (T.constant (7 :: Double))
  values (T.sum input) H.=== [7]
  values (T.mean input) H.=== [7]
  values (T.minimum input) H.=== [7]
  values (T.maximum input) H.=== [7]
  values (T.maximumAxes @'[] input) H.=== [7]

propEmptyRegion :: H.Property
propEmptyRegion = H.property $ do
  let input = T.zeros @'[2, 0, 3] :: T.Tensor '[2, 0, 3] Double
  values (T.sum input) H.=== [0]
  values (T.sumAxes @'[1] input) H.=== replicate 6 0
  values (T.sumAxes @'[2, 1] input) H.=== [0, 0]
  H.assert (all isNaN (values (T.mean input)))
  let result = values (T.meanAxes @'[1] input)
  length result H.=== 6
  H.assert (all isNaN result)

propEmptyOutput :: H.Property
propEmptyOutput = H.property $ do
  let input = T.zeros @'[0, 3] :: T.Tensor '[0, 3] Float
  values (T.sumAxes @'[1] input) H.=== []
  values (T.meanAxes @'[1] input) H.=== []
  values (T.minimumAxes @'[1] input) H.=== []
  values (T.maximumAxes @'[1] input) H.=== []
  values (T.minimumAxes @'[] input) H.=== []

propExtrema :: H.Property
propExtrema = H.property $ do
  values (T.maximum (T.fromList @'[2] [-5, -2 :: Int])) H.=== [-2]
  values (T.minimum (T.fromList @'[2] [5, 2 :: Int])) H.=== [2]
  values (T.minimum (T.fromList @'[2] [False, True])) H.=== [False]
  values (T.maximum (T.fromList @'[2] [False, True])) H.=== [True]
  values (T.minimum (T.full @'[2] (1 / 0 :: Double))) H.=== [1 / 0]
  values (T.maximum (T.full @'[2] (-(1 / 0) :: Float))) H.=== [-(1 / 0)]

propNaN :: H.Property
propNaN = H.property $ do
  let input = T.fromList @'[2, 3] [0 / 0, 2, 3, 4, 5, 0 / 0 :: Double]
      single = T.fromList @'[2] [1, 0 / 0 :: Float]
  H.assert (all isNaN (values (T.minimum input)))
  H.assert (all isNaN (values (T.maximum input)))
  H.assert (all isNaN (values (T.minimumAxes @'[1] input)))
  H.assert (all isNaN (values (T.maximumAxes @'[1] input)))
  H.assert (all isNaN (values (T.minimum single)))
  H.assert (all isNaN (values (T.maximum single)))

propComposition :: H.Property
propComposition = H.property $ do
  let input = T.fromList @'[2, 3] [1 .. 6 :: Int]
      total = T.sum (input * 2)
  values (T.fromGen @'[3] (\ix -> T.index @'[] total + T.coordinate @0 ix)) H.=== [42, 43, 44]
  values (T.sumAxes @'[1] (input + 1) * 2) H.=== [18, 36]

propBoolean :: H.Property
propBoolean = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) Gen.bool
  let input = T.fromList @'[2, 3] xs
      rows = [take 3 xs, drop 3 xs]
      cols = [[xs !! j, xs !! (3 + j)] | j <- [0 .. 2]]
  values (T.all input) H.=== [and xs]
  values (T.any input) H.=== [or xs]
  values (T.allAxes @'[1] input) H.=== map and rows
  values (T.anyAxes @'[0] input) H.=== map or cols
  values (T.allAxes @'[1, 0] input) H.=== [and xs]
  values (T.anyAxes @'[] input) H.=== xs

propBooleanEmpty :: H.Property
propBooleanEmpty = H.property $ do
  let input = T.full @'[2, 0, 3] False
  values (T.all input) H.=== [True]
  values (T.any input) H.=== [False]
  values (T.allAxes @'[1] input) H.=== replicate 6 True
  values (T.anyAxes @'[1] input) H.=== replicate 6 False
  values (T.allAxes @'[2] input) H.=== []
  values (T.any (T.scalar (T.constant True))) H.=== [True]

propNormalization :: H.Property
propNormalization = H.property $ do
  let input = T.fromList @'[2, 3] [1000, 1001, 1002, -1002, -1001, -1000 :: Double]
      weights = map exp [-2, -1, 0 :: Double]
      probabilities = map (/ sum weights) weights
      logs = map (subtract (log (sum weights))) [-2, -1, 0]
  approxList (values (T.softmax @1 input)) (probabilities ++ probabilities)
  approxList (values (T.logSoftmax @1 input)) (logs ++ logs)
  approxList (values (T.logSumExpAxes @'[1] input)) [1002 + log (sum weights), -1000 + log (sum weights)]
  approxList (values (T.logSumExp input)) [1002 + log (sum weights)]
  approxList (values (T.logSoftmax @0 (T.fromList @'[2] [0, -1000 :: Float]))) [0, -1000]

propNormalizationAxes :: H.Property
propNormalizationAxes = H.property $ do
  let input = T.fromList @'[2, 3] [0, 1, 2, 2, 3, 4 :: Double]
      lo = 1 / (1 + exp 2)
      hi = 1 - lo
      middle = T.fromList @'[2, 2, 2] [0, 1, 2, 3, 4, 5, 6, 7 :: Float]
  approxList (values (T.softmax @0 input)) (replicate 3 lo ++ replicate 3 hi)
  approxList (values (T.softmax @1 middle)) [realToFrac lo, realToFrac lo, realToFrac hi, realToFrac hi, realToFrac lo, realToFrac lo, realToFrac hi, realToFrac hi]
  approxList (values (T.sumAxes @'[1] (T.softmax @1 middle))) (replicate 4 1)
  approxList (values (T.map exp (T.logSoftmax @0 input))) (values (T.softmax @0 input))

propLogSumExpAxes :: H.Property
propLogSumExpAxes = H.property $ do
  let xs = [0 .. 11 :: Double]
      input = T.fromList @'[2, 3, 2] xs
      expected = [log (sum [exp (xs !! ((i * 3 + j) * 2 + k)) | i <- [0, 1], k <- [0, 1]]) | j <- [0 .. 2]]
  approxList (values (T.logSumExpAxes @'[2, 0] input)) expected
  values (T.logSumExpAxes @'[] input) H.=== xs
  approxList (values (T.logSumExp (T.scalar (T.constant (3 :: Double))))) [3]

propLogSumExpEdges :: H.Property
propLogSumExpEdges = H.property $ do
  let empty = T.zeros @'[2, 0] :: T.Tensor '[2, 0] Double
      input = T.fromList @'[4, 2] [1 / 0, 3, -(1 / 0), -(1 / 0), 0 / 0, 2, 0 / 0, 1 / 0 :: Double]
  values (T.logSumExp empty) H.=== [-(1 / 0)]
  values (T.logSumExpAxes @'[1] empty) H.=== replicate 2 (-(1 / 0))
  values (T.logSumExpAxes @'[1] (T.zeros @'[0, 2] :: T.Tensor '[0, 2] Float)) H.=== []
  case values (T.logSumExpAxes @'[1] input) of
    [positive, negative, nan, mixed] -> do
      positive H.=== 1 / 0
      negative H.=== -(1 / 0)
      H.assert (isNaN nan && isNaN mixed)
    _ -> H.failure
  values (T.logSumExp (T.full @'[2] (1 / 0 :: Float))) H.=== [1 / 0]
  values (T.logSumExp (T.full @'[2] (-(1 / 0) :: Float))) H.=== [-(1 / 0)]

propNormalizationEdges :: H.Property
propNormalizationEdges = H.property $ do
  approxList (values (T.softmax @1 (T.fromList @'[2, 1] [3, -5 :: Float]))) [1, 1]
  approxList (values (T.logSoftmax @1 (T.fromList @'[2, 1] [3, -5 :: Float]))) [0, 0]
  values (T.softmax @1 (T.zeros @'[0, 2] :: T.Tensor '[0, 2] Float)) H.=== []
  values (T.logSoftmax @1 (T.zeros @'[0, 2] :: T.Tensor '[0, 2] Float)) H.=== []
  H.assert (all isNaN (values (T.softmax @0 (T.fromList @'[2] [1 / 0, 0 :: Double]))))
  H.assert (all isNaN (values (T.logSoftmax @0 (T.fromList @'[2] [0 / 0, 0 :: Double]))))

firstPosition :: (Ord a) => ([a] -> a) -> [a] -> Maybe Int
firstPosition extremum xs = elemIndex (extremum xs) xs

propArgExtrema :: H.Property
propArgExtrema = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 12) (Gen.int (Range.linear (-10) 10))
  let input = T.fromList @'[3, 4] xs
      rows = [take 4 (drop (4 * i) xs) | i <- [0 .. 2]]
      cols = [[xs !! (4 * i + j) | i <- [0 .. 2]] | j <- [0 .. 3]]
  map Just (values (T.argmax @1 input)) H.=== map (firstPosition maximum) rows
  map Just (values (T.argmin @1 input)) H.=== map (firstPosition minimum) rows
  map Just (values (T.argmax @0 input)) H.=== map (firstPosition maximum) cols
  map Just (values (T.argmin @0 input)) H.=== map (firstPosition minimum) cols

propArgAxes :: H.Property
propArgAxes = H.property $ do
  let input = T.fromList @'[2, 3, 2] [1, 9, 5, 2, 5, 7, 8, 1, 4, 3, 0, 6 :: Int]
  values (T.argmax @1 input) H.=== [1, 0, 0, 2]
  values (T.argmin @1 input) H.=== [0, 1, 2, 0]
  values (T.argmax @0 input) H.=== [1, 0, 0, 1, 0, 0]
  values (T.argmin @2 input) H.=== [0, 1, 0, 1, 1, 0]
  let mask = T.fromList @'[2, 3] [False, True, True, False, False, False]
  values (T.argmax @1 mask) H.=== [1, 0]
  values (T.argmin @1 mask) H.=== [0, 0]

propArgSpecial :: H.Property
propArgSpecial = H.property $ do
  let input = T.fromList @'[4, 4] [3, 3, 1, 1, 0, 0 / 0, 2, 0 / 0, 0 / 0, 9, 0 / 0, 1, -(1 / 0), 1 / 0, 1 / 0, -(1 / 0) :: Double]
  values (T.argmax @1 input) H.=== [0, 1, 0, 1]
  values (T.argmin @1 input) H.=== [2, 1, 0, 0]
  values (T.argmax @0 (T.fromList @'[3] [-0.0, 0.0, -0.0 :: Float])) H.=== [0]
  values (T.argmin @0 (T.fromList @'[3] [-0.0, 0.0, -0.0 :: Float])) H.=== [0]
  let acrossBlocks = T.fromList @'[1024] (replicate 500 1 ++ [0 / 0] ++ replicate 522 2 ++ [0 / 0 :: Float])
  values (T.argmax @0 acrossBlocks) H.=== [500]
  values (T.argmin @0 acrossBlocks) H.=== [500]

propArgEmpty :: H.Property
propArgEmpty = H.property $ do
  values (T.argmax @1 (T.zeros @'[0, 3] :: T.Tensor '[0, 3] Int)) H.=== []
  values (T.argmin @1 (T.zeros @'[0, 3] :: T.Tensor '[0, 3] Float)) H.=== []
  values (T.argmax @1 (T.fromList @'[3, 1] [1, 2, 3 :: Double])) H.=== [0, 0, 0]
  values (T.argmin @0 (T.fromList @'[1] [True])) H.=== [0]
