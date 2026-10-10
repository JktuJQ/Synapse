{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Reduction.Operations (tests) where

import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh, T.DType a) => T.Tensor sh a -> [a]
values = T.toList . T.runCPU

element :: T.DType a => T.TensorElement a -> [a]
element = values . T.scalar

tests :: TestTree
tests = testGroup "operations"
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
    testProperty "reductions compose with element computations" (H.withTests 1 propComposition)
  ]

propFull :: H.Property
propFull = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let input = T.fromList @'[2, 3, 4] xs
  element (T.sum input) H.=== [sum xs]
  element (T.minimum input) H.=== [minimum xs]
  element (T.maximum input) H.=== [maximum xs]

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
  approxList (element (T.mean input)) [3.5]
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
  element (T.sum input) H.=== [7]
  element (T.mean input) H.=== [7]
  element (T.minimum input) H.=== [7]
  element (T.maximum input) H.=== [7]
  values (T.maximumAxes @'[] input) H.=== [7]

propEmptyRegion :: H.Property
propEmptyRegion = H.property $ do
  let input = T.zeros @'[2, 0, 3] :: T.Tensor '[2, 0, 3] Double
  element (T.sum input) H.=== [0]
  values (T.sumAxes @'[1] input) H.=== replicate 6 0
  values (T.sumAxes @'[2, 1] input) H.=== [0, 0]
  H.assert (all isNaN (element (T.mean input)))
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
  element (T.maximum (T.fromList @'[2] [-5, -2 :: Int])) H.=== [-2]
  element (T.minimum (T.fromList @'[2] [5, 2 :: Int])) H.=== [2]
  element (T.minimum (T.fromList @'[2] [False, True])) H.=== [False]
  element (T.maximum (T.fromList @'[2] [False, True])) H.=== [True]
  element (T.minimum (T.full @'[2] (1 / 0 :: Double))) H.=== [1 / 0]
  element (T.maximum (T.full @'[2] (-(1 / 0) :: Float))) H.=== [-(1 / 0)]

propNaN :: H.Property
propNaN = H.property $ do
  let input = T.fromList @'[2, 3] [0 / 0, 2, 3, 4, 5, 0 / 0 :: Double]
      single = T.fromList @'[2] [1, 0 / 0 :: Float]
  H.assert (all isNaN (element (T.minimum input)))
  H.assert (all isNaN (element (T.maximum input)))
  H.assert (all isNaN (values (T.minimumAxes @'[1] input)))
  H.assert (all isNaN (values (T.maximumAxes @'[1] input)))
  H.assert (all isNaN (element (T.minimum single)))
  H.assert (all isNaN (element (T.maximum single)))

propComposition :: H.Property
propComposition = H.property $ do
  let input = T.fromList @'[2, 3] [1 .. 6 :: Int]
      total = T.sum (input * 2)
  values (T.fromGen @'[3] (\ix -> total + T.coordinate @0 ix)) H.=== [42, 43, 44]
  values (T.sumAxes @'[1] (input + 1) * 2) H.=== [18, 36]
