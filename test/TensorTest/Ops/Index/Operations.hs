{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Index.Operations (tests) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (Nat)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Type as Internal
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "operations"
    [ testProperty "index operators read coordinates and elements" (H.withTests 1 propOperators),
      testProperty "index reads elements in row-major order" propIndex,
      testProperty "index composes with element expressions" (H.withTests 1 propIndexExpression),
      testProperty "mixed selections preserve axis order" propMixedSlice,
      testProperty "ranges retain singleton axes" (H.withTests 1 propRange),
      testProperty "strides round up the output length" (H.withTests 1 propRangeStep),
      testProperty "reverse changes the selected axis order" propRangeReverse,
      testProperty "RangeReverse reverses exactly the selected range" propReverseRange,
      testProperty "all At selections agree with index" (H.withTests 1 propAllAt),
      testProperty "slices support scalars and empty regions" (H.withTests 1 propEdges),
      testProperty "index supports non-numeric elements" (H.withTests 1 propBoolean),
      testProperty "index construction supports ranks above nine" (H.withTests 1 propHighRankIndex)
    ]

propIndex :: H.Property
propIndex = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3, 4] xs
  T.toList (T.run T.CPU (T.scalar (T.index @'[0, 0, 0] source))) H.=== take 1 xs
  T.toList (T.run T.CPU (T.scalar (T.index @'[1, 2, 3] source))) H.=== drop 23 xs

propIndexExpression :: H.Property
propIndexExpression = H.property $ do
  let source = T.fromList @'[2, 3] [0 .. 5 :: Int]
      value = T.index @'[1, 2] (source + 1)
      repeated = T.fromGen @'[3] (\ix -> value + T.coordinate @0 ix)
  T.toList (T.run T.CPU (T.scalar (value * 2))) H.=== [12]
  T.toList (T.run T.CPU repeated) H.=== [6, 7, 8]

propMixedSlice :: H.Property
propMixedSlice = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3, 4] xs
      result = T.run T.CPU (T.slice @'[ 'T.All, 'T.At 1, 'T.Range 0 4 2] source)
  A.arrayShape (Internal.unTensorData result) H.=== (A.Z A.:. (2 :: Int) A.:. (2 :: Int))
  T.toList result H.=== [xs !! ((i * 3 + 1) * 4 + k) | i <- [0, 1], k <- [0, 2]]

propRange :: H.Property
propRange = H.property $ do
  let source = T.fromList @'[3, 4] [0 .. 11 :: Int]
      result = T.run T.CPU (T.slice @'[ 'T.Range 1 2 1, 'T.Range 1 4 1] source)
  A.arrayShape (Internal.unTensorData result) H.=== (A.Z A.:. (1 :: Int) A.:. (3 :: Int))
  T.toList result H.=== [5, 6, 7]

propRangeStep :: H.Property
propRangeStep = H.property $ do
  let source = T.arange @8 (0 :: Int) 1
  T.toList (T.run T.CPU (T.slice @'[ 'T.Range 1 8 3] source)) H.=== [1, 4, 7]
  T.toList (T.run T.CPU (T.slice @'[ 'T.Range 1 7 10] source)) H.=== [1]
  T.toList (T.run T.CPU (T.slice @'[ 'T.Range 2 2 3] source)) H.=== []

propRangeReverse :: H.Property
propRangeReverse = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3, 4] xs
      reversed = T.slice @'[ 'T.RangeReverse 0 2 1, 'T.All, 'T.RangeReverse 0 4 1] source
  T.toList (T.run T.CPU reversed) H.=== [xs !! ((i * 3 + j) * 4 + k) | i <- [1, 0], j <- [0 .. 2], k <- [3, 2 .. 0]]
  T.toList (T.run T.CPU (T.slice @'[ 'T.RangeReverse 0 2 1, 'T.All, 'T.RangeReverse 0 4 1] reversed)) H.=== xs

propAllAt :: H.Property
propAllAt = H.property $ do
  let source = T.fromList @'[2, 3] [0 .. 5 :: Int]
      result = T.slice @'[ 'T.At 1, 'T.At 2] source
  T.toList (T.run T.CPU result) H.=== T.toList (T.run T.CPU (T.scalar (T.index @'[1, 2] source)))

propReverseRange :: H.Property
propReverseRange = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 8) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[8] xs
      reversed = T.run T.CPU (T.slice @'[ 'T.RangeReverse 1 7 2] source)
  A.arrayShape (Internal.unTensorData reversed) H.=== (A.Z A.:. (3 :: Int))
  T.toList reversed H.=== [xs !! 5, xs !! 3, xs !! 1]
  T.toList (T.run T.CPU (T.slice @'[ 'T.RangeReverse 1 8 3] source)) H.=== [xs !! 7, xs !! 4, xs !! 1]
  T.toList (T.run T.CPU (T.slice @'[ 'T.RangeReverse 1 7 10] source)) H.=== [xs !! 1]
  T.toList (T.run T.CPU (T.slice @'[ 'T.RangeReverse 4 4 2] source)) H.=== []
  T.toList (T.run T.CPU (T.slice @'[ 'T.RangeReverse 3 1 2] source)) H.=== []

propEdges :: H.Property
propEdges = H.property $ do
  let scalar = T.scalar (T.constant (7 :: Int))
      vector = T.arange @4 (0 :: Int) 1
      empty = T.zeros @'[2, 0, 3] :: T.Tensor '[2, 0, 3] Int
  T.toList (T.run T.CPU (T.scalar (T.index @'[] scalar))) H.=== [7]
  T.toList (T.run T.CPU (T.slice @'[] scalar)) H.=== [7]
  T.toList (T.run T.CPU (T.slice @'[ 'T.Range 4 4 1] vector)) H.=== []
  T.toList (T.run T.CPU (T.slice @'[ 'T.Range 3 1 1] vector)) H.=== []
  T.toList (T.run T.CPU (T.slice @'[ 'T.All, 'T.All, 'T.All] empty)) H.=== []
  T.toList (T.run T.CPU (T.slice @'[ 'T.All, 'T.RangeReverse 0 0 1, 'T.All] empty)) H.=== []

propBoolean :: H.Property
propBoolean = H.property $ do
  let source = T.fromList @'[2] [False, True]
  T.toList (T.run T.CPU (T.scalar (T.index @'[1] source))) H.=== [True]

propOperators :: H.Property
propOperators = H.property $ do
  let source = T.fromList @'[2, 3] [0 .. 5 :: Int]
      generated = T.fromGen @'[2, 3] (\ix -> (ix T.!! Proxy @0) * 10 + ix T.!! Proxy @1)
      copied = T.fromGen @'[2, 3] (source T.!!)
      scalar = T.scalar (T.constant True)
  T.toList (T.run T.CPU (T.scalar (source T.!! Proxy @'[1, 2]))) H.=== [5]
  T.toList (T.run T.CPU generated) H.=== [0, 1, 2, 10, 11, 12]
  T.toList (T.run T.CPU copied) H.=== [0 .. 5]
  T.toList (T.run T.CPU (T.scalar (scalar T.!! Proxy @('[] :: [Nat])))) H.=== [True]

propHighRankIndex :: H.Property
propHighRankIndex = H.property $ do
  let source = T.fromList @'[1, 1, 1, 1, 1, 1, 1, 1, 1, 4] [1 .. 4 :: Int]
      selected = T.slice @'[ 'T.All, 'T.All, 'T.All, 'T.All, 'T.All, 'T.All, 'T.All, 'T.All, 'T.All, 'T.RangeReverse 1 4 2] source
  T.toList (T.run T.CPU (T.scalar (T.index @'[0, 0, 0, 0, 0, 0, 0, 0, 0, 2] source))) H.=== [3]
  T.toList (T.run T.CPU selected) H.=== [4, 2]
