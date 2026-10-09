{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Elementwise (tests) where

import qualified Data.Array.Accelerate as A
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
    "elementwise"
    [ testProperty "map composition agrees with a single map" propMapComposition,
      testProperty "zipWith agrees with list combination" propZipWithLists,
      testProperty "map can change element type" propMap,
      testProperty "zip pairs different element types" (H.withTests 1 propZip),
      testProperty "zipWith supports distinct input and output types" (H.withTests 1 propZipWith),
      testProperty "element operations handle empty tensors and scalars" (H.withTests 1 propElementOperationEdges)
    ]

propMap :: H.Property
propMap = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let input = T.fromList @'[2, 3] xs
      output = T.map (\x -> T.fromIntegral x + 0.5) input :: T.Tensor '[2, 3] T.Float32
  T.toList (T.runCPU output) H.=== Prelude.map (\x -> fromIntegral x + 0.5) xs

propZip :: H.Property
propZip = H.property $ do
  let lhs = T.fromList @'[3] [1, 2, 3 :: Int]
      rhs = T.fromList @'[3] [True, False, True]
      paired = T.zip lhs rhs
      nested = T.zip paired lhs
  T.toList (T.runCPU paired) H.=== [(1, True), (2, False), (3, True)]
  T.toList (T.runCPU nested) H.=== [((1, True), 1), ((2, False), 2), ((3, True), 3)]

propZipWith :: H.Property
propZipWith = H.property $ do
  let lhs = T.fromList @'[3] [1, 2, 3 :: Int]
      rhs = T.fromList @'[3] [True, False, True]
      output =
        T.zipWith
          ( \x flag ->
              let converted = Internal.unTensorElement (T.fromIntegral x :: T.TensorElement T.Float32)
               in Internal.TensorElement (A.cond (Internal.unTensorElement flag) (converted + 0.5) (converted - 0.5))
          )
          lhs
          rhs
  T.toList (T.runCPU output) H.=== [1.5, 1.5, 3.5 :: T.Float32]

propElementOperationEdges :: H.Property
propElementOperationEdges = H.property $ do
  let empty = T.zeros @'[2, 0] :: T.Tensor '[2, 0] Int
      mapped = T.map (\_ -> T.constant True) empty
  T.toList (T.runCPU mapped) H.=== []
  T.toList (T.runCPU (T.zip empty mapped)) H.=== []
  T.toList (T.runCPU (T.zipWith (\x _ -> x + 1) empty mapped)) H.=== []
  T.toList (T.runCPU (T.map (+ 1) (T.scalar (2 :: Int)))) H.=== [3]
  T.toList (T.runCPU (T.zip (T.scalar (2 :: Int)) (T.scalar True))) H.=== [(2, True)]

propMapComposition :: H.Property
propMapComposition = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3] xs
      composed = T.map (+ 3) (T.map (* 2) source)
      single = T.map (\x -> x * 2 + 3) source
  T.toList (T.runCPU composed) H.=== map (\x -> x * 2 + 3) xs
  T.toList (T.runCPU single) H.=== T.toList (T.runCPU composed)

propZipWithLists :: H.Property
propZipWithLists = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let lhs = T.fromList @'[2, 3] xs
      rhs = T.fromList @'[2, 3] ys
      combined = T.zipWith (\x y -> x * 2 - y) lhs rhs
  T.toList (T.runCPU combined) H.=== zipWith (\x y -> x * 2 - y) xs ys
  T.toList (T.runCPU (T.zip lhs rhs)) H.=== zip xs ys
