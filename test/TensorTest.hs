{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest
  ( tests,
  )
where

import Control.Exception (ErrorCall, evaluate, try)
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
    "tensor"
    [ testProperty "shape exposes type-level dimensions" propShape,
      testProperty "shape supports rank above nine" propShapeAboveNine,
      testProperty "shapeSize multiplies dimensions" propShapeSize,
      testProperty "fromList round-trips through CPU backend" propFromListRoundTrip,
      testProperty "fromList rejects invalid element count" propFromListRejectsInvalidSize,
      testProperty "fromGen initializes by row-major index" propFromGen,
      testProperty "fromGen composes element expressions" propElementExpressions,
      testProperty "constructors handle empty and singleton shapes" propConstructorEdges,
      testProperty "Accelerate representations can be unpacked and reused" propAccelerateAccess,
      testProperty "full creates repeated values" propFull,
      testProperty "fill aliases full" propFill,
      testProperty "zeros creates zero values" propZeros,
      testProperty "ones creates one values" propOnes,
      testProperty "like constructors preserve shape" propLikeConstructors,
      testProperty "arange creates stepped values" propArange,
      testProperty "linspace includes endpoints" propLinspace,
      testProperty "logspace exponentiates linspace" propLogspace,
      testProperty "eye creates rectangular identity matrix" propEye,
      testProperty "identity creates square identity matrix" propIdentity,
      testProperty "Num instance is elementwise" propNumInstance,
      testProperty "map can change element type" propMap,
      testProperty "zip pairs different element types" propZip,
      testProperty "zipWith supports distinct input and output types" propZipWith,
      testProperty "element operations handle empty tensors and scalars" propElementOperationEdges,
      testProperty "Fractional instance is elementwise" propFractionalInstance,
      testProperty "Floating instance is elementwise" propFloatingInstance,
      testProperty "reshape preserves row-major order" propReshape,
      testProperty "broadcast expands scalars" propBroadcastScalar,
      testProperty "broadcast expands vectors across leading dimensions" propBroadcastVector,
      testProperty "broadcast expands singleton dimensions" propBroadcastSingletonDimension
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
  let tensor = T.fromGen @'[2, 3] id :: T.Tensor '[2, 3] Int
  T.toList (T.runCPU tensor) H.=== [0, 1, 2, 3, 4, 5]

propElementExpressions :: H.Property
propElementExpressions = H.property $ do
  let tensor = T.fromGen @'[2, 3] $ \i ->
        sqrt (T.fromIntegral i + T.constant (4 :: T.Float32)) / 2
  approxList (T.toList (T.runCPU tensor)) (map (\i -> sqrt (i + 4) / 2) [0 .. 5])

propConstructorEdges :: H.Property
propConstructorEdges = H.property $ do
  T.toList (T.runCPU (T.fromGen @'[] (\i -> i + 7))) H.=== [7]
  T.toList (T.runCPU (T.fromGen @'[2, 0] (\i -> i + 7))) H.=== []
  T.toList (T.runCPU (T.eye @2 @0 :: T.Tensor '[2, 0] Int)) H.=== []
  T.toList (T.runCPU (T.linspace @0 (2 :: T.Float32) 5)) H.=== []
  T.toList (T.runCPU (T.linspace @1 (2 :: T.Float32) 5)) H.=== [2]
  T.toList (T.runCPU (T.logspace @0 (2 :: T.Float32) 5 10)) H.=== []
  T.toList (T.runCPU (T.logspace @1 (2 :: T.Float32) 5 10)) H.=== [100]

propAccelerateAccess :: H.Property
propAccelerateAccess = H.property $ do
  let source = T.arange @3 (1 :: Int) 1
      modified =
        Internal.Tensor (A.map (+ 10) (Internal.unTensor source)) ::
          T.Tensor '[3] Int
      result = T.runCPU modified
      reused =
        Internal.Tensor (A.use (Internal.unTensorData result)) ::
          T.Tensor '[3] Int
      element = T.fromGen @'[3] $ \i ->
        Internal.TensorElement (Internal.unTensorElement i * 2)
  A.toList (Internal.unTensorData result) H.=== [11, 12, 13]
  T.toList (T.runCPU reused) H.=== [11, 12, 13]
  T.toList (T.runCPU element) H.=== [0, 2, 4]

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

propNumInstance :: H.Property
propNumInstance = H.property $ do
  let lhs = T.fromList @'[2, 2] [1, -2, 3, -4] :: T.Tensor '[2, 2] Int
      rhs = T.fromList @'[2, 2] [10, 20, 30, 40] :: T.Tensor '[2, 2] Int
      literal = 3 :: T.Tensor '[2, 2] Int
  T.toList (T.runCPU (lhs + rhs)) H.=== [11, 18, 33, 36]
  T.toList (T.runCPU (rhs - lhs)) H.=== [9, 22, 27, 44]
  T.toList (T.runCPU (lhs * rhs)) H.=== [10, -40, 90, -160]
  T.toList (T.runCPU (negate lhs)) H.=== [-1, 2, -3, 4]
  T.toList (T.runCPU (abs lhs)) H.=== [1, 2, 3, 4]
  T.toList (T.runCPU (signum lhs)) H.=== [1, -1, 1, -1]
  T.toList (T.runCPU literal) H.=== [3, 3, 3, 3]

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

propFractionalInstance :: H.Property
propFractionalInstance = H.property $ do
  let lhs = T.fromList @'[2] [8.0, 9.0] :: T.Tensor '[2] T.Float32
      rhs = T.fromList @'[2] [2.0, 3.0] :: T.Tensor '[2] T.Float32
      literal = 0.5 :: T.Tensor '[2] T.Float32
  T.toList (T.runCPU (lhs / rhs)) H.=== [4.0, 3.0]
  T.toList (T.runCPU (recip rhs)) H.=== [0.5, 1.0 / 3.0]
  T.toList (T.runCPU literal) H.=== [0.5, 0.5]

propFloatingInstance :: H.Property
propFloatingInstance = H.property $ do
  let tensor = T.fromList @'[2] [1.0, 4.0] :: T.Tensor '[2] T.Float32
      expInput = T.fromList @'[2] [0.0, 1.0] :: T.Tensor '[2] T.Float32
      logInput = T.fromList @'[2] [1.0, Prelude.exp 1.0] :: T.Tensor '[2] T.Float32
      piTensor = pi :: T.Tensor '[2] T.Float32
  approxList (T.toList (T.runCPU (sqrt tensor))) [1.0, 2.0]
  approxList (T.toList (T.runCPU (exp expInput))) [1.0, Prelude.exp 1.0]
  approxList (T.toList (T.runCPU (log logInput))) [0.0, 1.0]
  approxList (T.toList (T.runCPU piTensor)) [Prelude.pi, Prelude.pi]

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

approxList :: [T.Float32] -> [T.Float32] -> H.PropertyT IO ()
approxList actual expected =
  do
    length actual H.=== length expected
    H.assert $ and (zipWith (\x y -> abs (x - y) < 1e-5) actual expected)
