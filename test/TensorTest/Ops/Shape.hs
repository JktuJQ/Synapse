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
    [ testProperty "concat joins outer and inner axes" propConcat,
      testProperty "concat joins a middle axis" (H.withTests 1 propConcatMiddle),
      testProperty "stack inserts the requested axis" propStack,
      testProperty "joins support empty dimensions and scalars" (H.withTests 1 propJoinEdges),
      testProperty "joins support non-numeric elements" (H.withTests 1 propJoinBoolean),
      testProperty "transpose swaps matrix axes" propTranspose,
      testProperty "transpose preserves multiple batch dimensions" (H.withTests 1 propBatchedTranspose),
      testProperty "permutations preserve values and invert correctly" propPermute,
      testProperty "permutations support scalars, empty tensors, and boolean elements" (H.withTests 1 propPermutationEdges),
      testProperty "squeeze removes the selected singleton axis" (H.withTests 1 propSqueeze),
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

propTranspose :: H.Property
propTranspose = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3] xs
      result = T.runCPU (T.transpose source)
  A.arrayShape (Internal.unTensorData result) H.=== (A.Z A.:. (3 :: Int) A.:. (2 :: Int))
  T.toList result H.=== [xs !! (i * 3 + j) | j <- [0 .. 2], i <- [0, 1]]
  T.toList (T.runCPU (T.transpose (T.transpose source))) H.=== xs

propBatchedTranspose :: H.Property
propBatchedTranspose = H.property $ do
  let xs = [0 .. 47 :: Int]
      source = T.fromList @'[2, 2, 3, 4] xs
      transposed = T.transpose source
  T.toList (T.runCPU transposed) H.=== [xs !! ((batch * 3 + row) * 4 + col) | batch <- [0 .. 3], col <- [0 .. 3], row <- [0 .. 2]]
  T.toList (T.runCPU (T.transpose transposed)) H.=== xs
  let matrix = T.fromList @'[2, 3] [1 .. 6 :: Int]
  T.toList (T.runCPU (T.matmul matrix (T.transpose matrix))) H.=== [14, 32, 32, 77]

propPermute :: H.Property
propPermute = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3, 4] xs
      permuted = T.permuteAxes @'[1, 2, 0] source
  T.toList (T.runCPU permuted) H.=== [xs !! ((i * 3 + j) * 4 + k) | j <- [0 .. 2], k <- [0 .. 3], i <- [0, 1]]
  T.toList (T.runCPU (T.permuteAxes @'[2, 0, 1] permuted)) H.=== xs
  T.toList (T.runCPU (T.permuteAxes @'[0, 1, 2] source)) H.=== xs
  T.toList (T.runCPU (T.permuteAxes @'[2, 1, 0] source)) H.=== [xs !! ((i * 3 + j) * 4 + k) | k <- [0 .. 3], j <- [0 .. 2], i <- [0, 1]]

propPermutationEdges :: H.Property
propPermutationEdges = H.property $ do
  let scalar = T.scalar (T.constant True)
      empty = T.zeros @'[2, 0, 3] :: T.Tensor '[2, 0, 3] Int
      flags = T.fromList @'[2, 3] [True, False, False, True, True, False]
  T.toList (T.runCPU (T.permuteAxes @'[] scalar)) H.=== [True]
  T.toList (T.runCPU (T.permuteAxes @'[2, 0, 1] empty)) H.=== []
  T.toList (T.runCPU (T.transpose empty)) H.=== []
  T.toList (T.runCPU (T.transpose flags)) H.=== [True, True, False, True, False, False]
  T.toList (T.runCPU (T.permuteAxes @'[0] (T.fromList @'[2] [False, True]))) H.=== [False, True]

propConcat :: H.Property
propConcat = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 3) (Gen.int (Range.linear (-100) 100))
  let lhs = T.fromList @'[2, 3] xs
      rows = T.fromList @'[1, 3] ys
      cols = T.fromList @'[2, 1] (take 2 ys)
      appended = T.runCPU (T.concat @0 lhs rows)
  A.arrayShape (Internal.unTensorData appended) H.=== (A.Z A.:. (3 :: Int) A.:. (3 :: Int))
  T.toList appended H.=== xs ++ ys
  T.toList (T.runCPU (T.concat @1 lhs cols)) H.=== take 3 xs ++ take 1 ys ++ drop 3 xs ++ take 1 (drop 1 ys)

propConcatMiddle :: H.Property
propConcatMiddle = H.property $ do
  let lhs = T.fromList @'[2, 1, 3] [0 .. 5 :: Int]
      rhs = T.fromList @'[2, 2, 3] [10 .. 21 :: Int]
  T.toList (T.runCPU (T.concat @1 lhs rhs)) H.=== [0, 1, 2, 10, 11, 12, 13, 14, 15, 3, 4, 5, 16, 17, 18, 19, 20, 21]

propStack :: H.Property
propStack = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let lhs = T.fromList @'[2, 3] xs
      rhs = T.fromList @'[2, 3] ys
      interleaved = concat (zipWith (\x y -> [x, y]) xs ys)
      middle = T.runCPU (T.stack @1 lhs rhs)
  T.toList (T.runCPU (T.stack @0 lhs rhs)) H.=== xs ++ ys
  A.arrayShape (Internal.unTensorData middle) H.=== (A.Z A.:. (2 :: Int) A.:. (2 :: Int) A.:. (3 :: Int))
  T.toList middle H.=== take 3 xs ++ take 3 ys ++ drop 3 xs ++ drop 3 ys
  T.toList (T.runCPU (T.stack @2 lhs rhs)) H.=== interleaved

propJoinEdges :: H.Property
propJoinEdges = H.property $ do
  let empty = T.zeros @'[2, 0, 3] :: T.Tensor '[2, 0, 3] Int
      input = T.fromList @'[2, 1, 3] [0 .. 5 :: Int]
  T.toList (T.runCPU (T.concat @1 empty input)) H.=== [0 .. 5]
  T.toList (T.runCPU (T.concat @1 input empty)) H.=== [0 .. 5]
  T.toList (T.runCPU (T.concat @1 empty empty)) H.=== []
  T.toList (T.runCPU (T.concat @0 empty empty)) H.=== []
  T.toList (T.runCPU (T.stack @1 empty empty)) H.=== []
  T.toList (T.runCPU (T.stack @0 (T.scalar (T.constant (3 :: Int))) (T.scalar 7))) H.=== [3, 7]

propJoinBoolean :: H.Property
propJoinBoolean = H.property $ do
  let lhs = T.fromList @'[2] [True, False]
      rhs = T.fromList @'[2] [False, True]
  T.toList (T.runCPU (T.concat @0 lhs rhs)) H.=== [True, False, False, True]
  T.toList (T.runCPU (T.stack @1 lhs rhs)) H.=== [True, False, False, True]
