{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Shape (tests) where

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
    "shape operations"
    [ testProperty "squeeze removes the selected singleton axis" (H.withTests 1 propSqueeze),
      testProperty "unsqueeze supports every insertion position" (H.withTests 1 propUnsqueezePositions),
      testProperty "unsqueeze and squeeze round-trip" propUnsqueezeSqueeze,
      testProperty "axis operations preserve non-numeric elements" (H.withTests 1 propNonNumericAxes),
      testProperty "squeeze preserves other singleton axes" (H.withTests 1 propSqueezeOneAxis),
      testProperty "axis operations support scalars and empty tensors" (H.withTests 1 propAxisEdges),
      testProperty "reshape preserves row-major order" propReshape,
      testProperty "reshape round-trips through a higher rank" propReshapeRoundTrip,
      testProperty "reshape supports scalars and empty tensors" (H.withTests 1 propReshapeEdges),
      testProperty "broadcast expands multiple axes" propBroadcastMultipleAxes,
      testProperty "broadcast preserves identical shapes" propBroadcastIdentity,
      testProperty "broadcast supports empty targets" (H.withTests 1 propBroadcastEmpty),
      testProperty "broadcast expands scalars" (H.withTests 1 propBroadcastScalar),
      testProperty "broadcast expands vectors across leading dimensions" (H.withTests 1 propBroadcastVector),
      testProperty "broadcast expands singleton dimensions" (H.withTests 1 propBroadcastSingletonDimension)
    ]

propReshape :: H.Property
propReshape = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.float (Range.linearFracFrom 0 (-100) 100))
  let tensor = T.fromList @'[2, 3] xs :: T.Tensor '[2, 3] T.Float32
      reshaped = T.reshape @'[3, 2] tensor
  T.toList (T.runCPU reshaped) H.=== xs

propBroadcastScalar :: H.Property
propBroadcastScalar = H.property $ do
  let tensor = T.scalar (T.constant (3.0 :: T.Float32))
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

propReshapeRoundTrip :: H.Property
propReshapeRoundTrip = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3] xs
      reshaped = T.reshape @'[3, 1, 2] source
  A.arrayShape (Internal.unTensorData (T.runCPU reshaped)) H.=== (A.Z A.:. (3 :: Int) A.:. (1 :: Int) A.:. (2 :: Int))
  T.toList (T.runCPU (T.reshape @'[2, 3] reshaped)) H.=== xs

propReshapeEdges :: H.Property
propReshapeEdges = H.property $ do
  T.toList (T.runCPU (T.reshape @'[1, 1] (T.scalar (T.constant (7 :: Int))))) H.=== [7]
  T.toList (T.runCPU (T.reshape @'[] (T.full @'[1, 1] (7 :: Int)))) H.=== [7]
  T.toList (T.runCPU (T.reshape @'[0] (T.zeros @'[2, 0, 3] :: T.Tensor '[2, 0, 3] Int))) H.=== []

propBroadcastMultipleAxes :: H.Property
propBroadcastMultipleAxes = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 1, 3] xs
      result = T.broadcast @'[4, 2, 5, 3] source
      expected = concat [concat (replicate 5 row) | _ <- [1 .. 4 :: Int], row <- [take 3 xs, drop 3 xs]]
  T.toList (T.runCPU result) H.=== expected

propBroadcastIdentity :: H.Property
propBroadcastIdentity = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  T.toList (T.runCPU (T.broadcast @'[2, 3] (T.fromList @'[2, 3] xs))) H.=== xs

propBroadcastEmpty :: H.Property
propBroadcastEmpty = H.property $ do
  T.toList (T.runCPU (T.broadcast @'[2, 0, 3] (T.ones @'[1, 3] :: T.Tensor '[1, 3] Int))) H.=== []

propSqueeze :: H.Property
propSqueeze = H.property $ do
  let middle = T.squeeze @1 (T.fromList @'[2, 1, 3] [0 .. 5 :: Int])
      first = T.squeeze @0 (T.fromList @'[1, 2, 3] [0 .. 5 :: Int])
      lastAxis = T.squeeze @2 (T.fromList @'[2, 3, 1] [0 .. 5 :: Int])
  A.arrayShape (Internal.unTensorData (T.runCPU middle)) H.=== (A.Z A.:. (2 :: Int) A.:. (3 :: Int))
  T.toList (T.runCPU middle) H.=== [0 .. 5]
  T.toList (T.runCPU first) H.=== [0 .. 5]
  T.toList (T.runCPU lastAxis) H.=== [0 .. 5]

propUnsqueezePositions :: H.Property
propUnsqueezePositions = H.property $ do
  let source = T.fromList @'[2, 3] [0 .. 5 :: Int]
      first = T.runCPU (T.unsqueeze @0 source)
      middle = T.runCPU (T.unsqueeze @1 source)
      lastAxis = T.runCPU (T.unsqueeze @2 source)
  A.arrayShape (Internal.unTensorData first) H.=== (A.Z A.:. (1 :: Int) A.:. (2 :: Int) A.:. (3 :: Int))
  A.arrayShape (Internal.unTensorData middle) H.=== (A.Z A.:. (2 :: Int) A.:. (1 :: Int) A.:. (3 :: Int))
  A.arrayShape (Internal.unTensorData lastAxis) H.=== (A.Z A.:. (2 :: Int) A.:. (3 :: Int) A.:. (1 :: Int))
  T.toList first H.=== [0 .. 5]
  T.toList middle H.=== [0 .. 5]
  T.toList lastAxis H.=== [0 .. 5]

propUnsqueezeSqueeze :: H.Property
propUnsqueezeSqueeze = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3] xs
  T.toList (T.runCPU (T.squeeze @1 (T.unsqueeze @1 source))) H.=== xs

propNonNumericAxes :: H.Property
propNonNumericAxes = H.property $ do
  let source = T.fromList @'[2, 3] [True, False, True, False, True, False]
  T.toList (T.runCPU (T.squeeze @2 (T.unsqueeze @2 source))) H.=== [True, False, True, False, True, False]

propSqueezeOneAxis :: H.Property
propSqueezeOneAxis = H.property $ do
  let result = T.runCPU (T.squeeze @1 (T.full @'[1, 1, 2, 1] (7 :: Int)))
  A.arrayShape (Internal.unTensorData result) H.=== (A.Z A.:. (1 :: Int) A.:. (2 :: Int) A.:. (1 :: Int))
  T.toList result H.=== [7, 7]

propAxisEdges :: H.Property
propAxisEdges = H.property $ do
  let scalar = T.scalar (T.constant (7 :: Int))
      empty = T.zeros @'[2, 0, 3] :: T.Tensor '[2, 0, 3] Int
  T.toList (T.runCPU (T.squeeze @0 (T.unsqueeze @0 scalar))) H.=== [7]
  T.toList (T.runCPU (T.squeeze @1 (T.unsqueeze @1 empty))) H.=== []
