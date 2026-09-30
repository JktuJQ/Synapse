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
