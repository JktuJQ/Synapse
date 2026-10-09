{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Construction (tests) where

import qualified Hedgehog as H
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "construction"
    [ testProperty "arange supports negative and zero steps" (H.withTests 1 propArangeSteps),
      testProperty "linspace supports descending endpoints" (H.withTests 1 propDescendingLinspace),
      testProperty "eye supports tall matrices" (H.withTests 1 propTallEye),
      testProperty "fromGen initializes by row-major index" (H.withTests 1 propFromGen),
      testProperty "fromGen composes element expressions" (H.withTests 1 propElementExpressions),
      testProperty "constructors handle empty and singleton shapes" (H.withTests 1 propConstructorEdges),
      testProperty "full creates repeated values" (H.withTests 1 propFull),
      testProperty "fill aliases full" (H.withTests 1 propFill),
      testProperty "zeros creates zero values" (H.withTests 1 propZeros),
      testProperty "ones creates one values" (H.withTests 1 propOnes),
      testProperty "like constructors preserve shape" (H.withTests 1 propLikeConstructors),
      testProperty "arange creates stepped values" (H.withTests 1 propArange),
      testProperty "linspace includes endpoints" (H.withTests 1 propLinspace),
      testProperty "logspace exponentiates linspace" (H.withTests 1 propLogspace),
      testProperty "eye creates rectangular identity matrix" (H.withTests 1 propEye),
      testProperty "identity creates square identity matrix" (H.withTests 1 propIdentity)
    ]

propArangeSteps :: H.Property
propArangeSteps = H.property $ do
  T.toList (T.runCPU (T.arange @4 (5 :: Int) (-2))) H.=== [5, 3, 1, -1]
  T.toList (T.runCPU (T.arange @4 (5 :: Int) 0)) H.=== [5, 5, 5, 5]
  approxList (T.toList (T.runCPU (T.arange @4 (0.5 :: T.Float64) 0.25))) [0.5, 0.75, 1, 1.25]

propDescendingLinspace :: H.Property
propDescendingLinspace = H.property $ do
  approxList (T.toList (T.runCPU (T.linspace @4 (3 :: T.Float64) 0))) [3, 2, 1, 0]

propTallEye :: H.Property
propTallEye = H.property $ do
  T.toList (T.runCPU (T.eye @3 @2 :: T.Tensor '[3, 2] Int)) H.=== [1, 0, 0, 1, 0, 0]

propFromGen :: H.Property
propFromGen = H.property $ do
  let tensor = T.fromGen @'[2, 3] id :: T.Tensor '[2, 3] Int
  T.toList (T.runCPU tensor) H.=== [0, 1, 2, 3, 4, 5]

propElementExpressions :: H.Property
propElementExpressions = H.property $ do
  let tensor = T.fromGen @'[2, 3] $ \i ->
        sqrt (T.fromIntegral i + T.constant (4 :: T.Float32)) / 2
  approxList (T.toList (T.runCPU tensor)) (map (\i -> sqrt (i + 4) / 2) [0 .. 5])

propConstructorEdges :: H.Property
propConstructorEdges = H.property $ do
  T.toList (T.runCPU (T.fromGen @'[] (+ 7))) H.=== [7]
  T.toList (T.runCPU (T.fromGen @'[2, 0] (+ 7))) H.=== []
  T.toList (T.runCPU (T.eye @2 @0 :: T.Tensor '[2, 0] Int)) H.=== []
  T.toList (T.runCPU (T.linspace @0 (2 :: T.Float32) 5)) H.=== []
  T.toList (T.runCPU (T.linspace @1 (2 :: T.Float32) 5)) H.=== [2]
  T.toList (T.runCPU (T.logspace @0 (2 :: T.Float32) 5 10)) H.=== []
  T.toList (T.runCPU (T.logspace @1 (2 :: T.Float32) 5 10)) H.=== [100]

propFull :: H.Property
propFull = H.property $ do
  let tensor = T.full @'[2, 3] (7 :: Int)
  T.toList (T.runCPU tensor) H.=== replicate 6 7

propFill :: H.Property
propFill = H.property $ do
  let tensor = T.fill @'[2, 3] (7 :: Int)
  T.toList (T.runCPU tensor) H.=== replicate 6 7

propZeros :: H.Property
propZeros = H.property $ do
  let tensor = T.zeros @'[2, 3] :: T.Tensor '[2, 3] Int
  T.toList (T.runCPU tensor) H.=== replicate 6 0

propOnes :: H.Property
propOnes = H.property $ do
  let tensor = T.ones @'[2, 3] :: T.Tensor '[2, 3] Int
  T.toList (T.runCPU tensor) H.=== replicate 6 1

propLikeConstructors :: H.Property
propLikeConstructors = H.property $ do
  let source = T.fromList @'[2, 2] [1, 2, 3, 4] :: T.Tensor '[2, 2] Int
      fullLike = T.fullLike source (5 :: Int)
      zerosLike = T.zerosLike source :: T.Tensor '[2, 2] Int
      onesLike = T.onesLike source :: T.Tensor '[2, 2] Int
  T.toList (T.runCPU fullLike) H.=== [5, 5, 5, 5]
  T.toList (T.runCPU zerosLike) H.=== [0, 0, 0, 0]
  T.toList (T.runCPU onesLike) H.=== [1, 1, 1, 1]

propArange :: H.Property
propArange = H.property $ do
  let tensor = T.arange @5 (2 :: Int) 3
  T.toList (T.runCPU tensor) H.=== [2, 5, 8, 11, 14]

propLinspace :: H.Property
propLinspace = H.property $ do
  let tensor = T.linspace @3 (0 :: T.Float32) 1
  T.toList (T.runCPU tensor) H.=== [0.0, 0.5, 1.0]

propLogspace :: H.Property
propLogspace = H.property $ do
  let tensor = T.logspace @3 (0 :: T.Float32) 2 10
  T.toList (T.runCPU tensor) H.=== [1.0, 10.0, 100.0]

propEye :: H.Property
propEye = H.property $ do
  let tensor = T.eye @2 @3 :: T.Tensor '[2, 3] Int
  T.toList (T.runCPU tensor) H.=== [1, 0, 0, 0, 1, 0]

propIdentity :: H.Property
propIdentity = H.property $ do
  let tensor = T.identity @3 :: T.Tensor '[3, 3] Int
  T.toList (T.runCPU tensor) H.=== [1, 0, 0, 0, 1, 0, 0, 0, 1]
