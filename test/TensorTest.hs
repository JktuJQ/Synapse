{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest
  ( tests
  ) where

import Control.Exception (ErrorCall, evaluate, try)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T

tests :: TestTree
tests =
  testGroup
    "tensor"
    [ testProperty "shape exposes type-level dimensions" propShape
    , testProperty "shape supports rank above nine" propShapeAboveNine
    , testProperty "shapeSize multiplies dimensions" propShapeSize
    , testProperty "fromList round-trips through CPU backend" propFromListRoundTrip
    , testProperty "fromList rejects invalid element count" propFromListRejectsInvalidSize
    , testProperty "fromGen initializes by row-major index" propFromGen
    , testProperty "full creates repeated values" propFull
    , testProperty "fill aliases full" propFill
    , testProperty "zeros creates zero values" propZeros
    , testProperty "ones creates one values" propOnes
    , testProperty "like constructors preserve shape" propLikeConstructors
    , testProperty "arange creates stepped values" propArange
    , testProperty "linspace includes endpoints" propLinspace
    , testProperty "logspace exponentiates linspace" propLogspace
    , testProperty "eye creates rectangular identity matrix" propEye
    , testProperty "identity creates square identity matrix" propIdentity
    , testProperty "reshape preserves row-major order" propReshape
    , testProperty "broadcast expands scalars" propBroadcastScalar
    , testProperty "broadcast expands vectors across leading dimensions" propBroadcastVector
    , testProperty "broadcast expands singleton dimensions" propBroadcastSingletonDimension
    ]

propShape :: H.Property
propShape = H.property $ do
  T.shape @'[2, 3, 4] H.=== [2, 3, 4]

propShapeAboveNine :: H.Property
propShapeAboveNine = H.property $ do
  T.shape @'[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] H.=== [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

propShapeSize :: H.Property
propShapeSize = H.property $ do
  T.shapeSize @'[2, 3, 4] H.=== 24

propFromListRoundTrip :: H.Property
propFromListRoundTrip = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.float (Range.linearFracFrom 0 (-100) 100))
  let tensor = T.fromList @'[2, 3] xs :: T.Tensor '[2, 3] T.Float32
  T.toList (T.runCPU tensor) H.=== xs

propFromListRejectsInvalidSize :: H.Property
propFromListRejectsInvalidSize = H.property $ do
  result <- H.evalIO $ try @ErrorCall $ evaluate $ T.fromList @'[2, 3] [1.0, 2.0 :: T.Float32]
  case result of
    Left _ -> H.success
    Right _ -> H.failure

propFromGen :: H.Property
propFromGen = H.property $ do
  let tensor = T.fromGen @'[2, 3] fromIntegral :: T.Tensor '[2, 3] Int
  T.toList (T.runCPU tensor) H.=== [0, 1, 2, 3, 4, 5]

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

propReshape :: H.Property
propReshape = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.float (Range.linearFracFrom 0 (-100) 100))
  let tensor = T.fromList @'[2, 3] xs :: T.Tensor '[2, 3] T.Float32
      reshaped = T.reshape @'[3, 2] tensor
  T.toList (T.runCPU reshaped) H.=== xs

propBroadcastScalar :: H.Property
propBroadcastScalar = H.property $ do
  let tensor = T.scalar (3.0 :: T.Float32)
      broadcasted = T.broadcast @'[2, 2] tensor
  T.toList (T.runCPU broadcasted) H.=== [3.0, 3.0, 3.0, 3.0]

propBroadcastVector :: H.Property
propBroadcastVector = H.property $ do
  let tensor = T.fromList @'[3] [1.0, 2.0, 3.0] :: T.Tensor '[3] T.Float32
      broadcasted = T.broadcast @'[2, 3] tensor
  T.toList (T.runCPU broadcasted) H.=== [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

propBroadcastSingletonDimension :: H.Property
propBroadcastSingletonDimension = H.property $ do
  let tensor = T.fromList @'[1, 3] [1.0, 2.0, 3.0] :: T.Tensor '[1, 3] T.Float32
      broadcasted = T.broadcast @'[2, 3] tensor
  T.toList (T.runCPU broadcasted) H.=== [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
