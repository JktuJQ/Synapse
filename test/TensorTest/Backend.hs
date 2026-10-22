{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Backend (tests) where

import qualified Data.Array.Accelerate as A
import qualified Hedgehog as H
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Type as Internal
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "backend"
    [ testProperty "run executes closed tensor computations on the CPU" (H.withTests 1 propCPUDispatch),
      testProperty "Accelerate representations can be unpacked and reused" (H.withTests 1 propAccelerateAccess),
      testProperty "run1 reuses a function with changing input values" (H.withTests 1 propRun1),
      testProperty "run1 permits changing output shape and element type" (H.withTests 1 propRun1Types),
      testProperty "run2 accepts both changing inputs" (H.withTests 1 propRun2),
      testProperty "prepared functions support scalars and empty tensors" (H.withTests 1 propPreparedEdges),
      testProperty "run3 accepts three changing tensor inputs" (H.withTests 1 propRun3),
      testProperty "run3 accepts different input element types" (H.withTests 1 propRun3Types),
      testProperty "run4 preserves argument order and accepts new inputs" (H.withTests 1 propRun4),
      testProperty "run5 preserves argument order and accepts new inputs" (H.withTests 1 propRun5),
      testProperty "run6 preserves argument order and accepts new inputs" (H.withTests 1 propRun6),
      testProperty "run7 preserves argument order and accepts new inputs" (H.withTests 1 propRun7),
      testProperty "run8 preserves argument order and accepts new inputs" (H.withTests 1 propRun8),
      testProperty "run9 preserves argument order and accepts new inputs" (H.withTests 1 propRun9),
      testProperty "run4 supports partial application with heterogeneous inputs" (H.withTests 1 propRun4Types),
      testProperty "run9 returns nested results with independent shapes and types" (H.withTests 1 propRun9Types),
      testProperty "run materializes nested results with independent shapes and element types" (H.withTests 1 propNestedComputation),
      testProperty "run1 infers tuple outputs from its materialized signature" (H.withTests 1 propRun1Outputs),
      testProperty "run2 returns a matrix and a scalar in one result" (H.withTests 1 propRun2Outputs),
      testProperty "run3 returns logits and statistics for changing inputs" (H.withTests 1 propRun3Outputs),
      testProperty "tuple outputs preserve empty tensors and scalars" (H.withTests 1 propComputationEdges),
      testProperty "execution supports 2 independent outputs" (H.withTests 1 propOutputs2),
      testProperty "execution supports 3 independent outputs" (H.withTests 1 propOutputs3),
      testProperty "execution supports 4 independent outputs" (H.withTests 1 propOutputs4),
      testProperty "execution supports 5 independent outputs" (H.withTests 1 propOutputs5),
      testProperty "execution supports 6 independent outputs" (H.withTests 1 propOutputs6),
      testProperty "execution supports 7 independent outputs" (H.withTests 1 propOutputs7),
      testProperty "execution supports 8 independent outputs" (H.withTests 1 propOutputs8),
      testProperty "execution supports 9 independent outputs" (H.withTests 1 propOutputs9)
    ]

propRun4Types :: H.Property
propRun4Types = H.property $ do
  let execute :: T.TensorData '[2, 3] Int -> T.TensorData '[3, 2] Int -> T.TensorData '[2] Double -> T.TensorData '[2, 2] Bool -> T.TensorData '[2, 2] Double
      execute = T.run4 T.CPU (\x w b m -> T.select m (T.map T.fromIntegral (T.matmul x w) + T.broadcast @'[2, 2] b) 0)
      lhs = T.run T.CPU (T.fromList @'[2, 3] [1 .. 6 :: Int])
      rhs = T.run T.CPU (T.fromList @'[3, 2] [1 .. 6 :: Int])
      bias = T.run T.CPU (T.fromList @'[2] [0.5, 1.5 :: Double])
      mask = T.run T.CPU (T.fromList @'[2, 2] [True, False, False, True])
      applyMask = execute lhs rhs bias
      nextStage = T.run1 T.CPU T.transpose
  T.toList (applyMask mask) H.=== [22.5, 0, 0, 65.5]
  T.toList (applyMask (T.run T.CPU (T.full @'[2, 2] True))) H.=== [22.5, 29.5, 49.5, 65.5]
  T.toList (nextStage (applyMask mask)) H.=== [22.5, 0, 0, 65.5]
  T.toList (applyMask mask) H.=== [22.5, 0, 0, 65.5]

propRun9Types :: H.Property
propRun9Types = H.property $ do
  let execute :: T.TensorData '[] Int -> T.TensorData '[2] Double -> T.TensorData '[2, 1] Bool -> T.TensorData '[0] Int -> T.TensorData '[1, 2] Float -> T.TensorData '[3] Int -> T.TensorData '[] Bool -> T.TensorData '[1] Double -> T.TensorData '[2, 0] Float -> ((T.TensorData '[] Int, T.TensorData '[2] Double, T.TensorData '[2, 1] Bool), (T.TensorData '[0] Int, T.TensorData '[1, 2] Float, T.TensorData '[3] Int), (T.TensorData '[] Bool, T.TensorData '[1] Double, T.TensorData '[2, 0] Float))
      execute = T.run9 T.CPU (\a b c d e f g h i -> ((a, b, c), (d, e, f), (g, h, i)))
      scalar = T.run T.CPU (T.full @'[] (7 :: Int))
      vector = T.run T.CPU (T.fromList @'[2] [1.5, 2.5 :: Double])
      mask = T.run T.CPU (T.fromList @'[2, 1] [True, False])
      empty = T.run T.CPU (T.fromList @'[0] ([] :: [Int]))
      matrix = T.run T.CPU (T.fromList @'[1, 2] [3, 4 :: Float])
      indices = T.run T.CPU (T.fromList @'[3] [2, 1, 0 :: Int])
      flag = T.run T.CPU (T.full @'[] False)
      singleton = T.run T.CPU (T.full @'[1] (8.5 :: Double))
      emptyMatrix = T.run T.CPU (T.fromList @'[2, 0] ([] :: [Float]))
      check input expected = do
        let ((a, b, c), (d, e, f), (g, h, i)) = execute input vector mask empty matrix indices flag singleton emptyMatrix
        T.toList a H.=== [expected]
        T.toList b H.=== [1.5, 2.5]
        T.toList c H.=== [True, False]
        T.toList d H.=== []
        T.toList e H.=== [3, 4]
        T.toList f H.=== [2, 1, 0]
        T.toList g H.=== [False]
        T.toList h H.=== [8.5]
        T.toList i H.=== []
  check scalar 7
  check (T.run T.CPU (T.full @'[] (-3 :: Int))) (-3)
  check scalar 7

propCPUDispatch :: H.Property
propCPUDispatch = H.property $ do
  let tensor = T.arange @4 (1 :: Int) 2
  T.toList (T.run T.CPU tensor) H.=== [1, 3, 5, 7]
  T.toList (T.run T.CPU (T.sum tensor)) H.=== [16]

propAccelerateAccess :: H.Property
propAccelerateAccess = H.property $ do
  let source = T.arange @3 (1 :: Int) 1
      modified =
        Internal.Tensor (A.map (+ 10) (Internal.unTensor source)) ::
          T.Tensor '[3] Int
      result = T.run T.CPU modified
      reused =
        T.fromData result ::
          T.Tensor '[3] Int
      element = T.fromGen @'[3] $ \i ->
        Internal.TensorElement (Internal.unTensorElement (T.coordinate @0 i) * 2)
  A.toList (Internal.unTensorData result) H.=== [11, 12, 13]
  T.toList (T.run T.CPU reused) H.=== [11, 12, 13]
  T.toList (T.run T.CPU element) H.=== [0, 2, 4]

propRun1 :: H.Property
propRun1 = H.property $ do
  let execute = T.run1 T.CPU (\x -> x * 2 + 1) :: T.TensorData '[3] Int -> T.TensorData '[3] Int
      first = T.run T.CPU (T.fromList @'[3] [1, 2, 3 :: Int])
      second = T.run T.CPU (T.fromList @'[3] [-2, 0, 4 :: Int])
  T.toList (execute first) H.=== [3, 5, 7]
  T.toList (execute second) H.=== [-3, 1, 9]
  T.toList (execute first) H.=== [3, 5, 7]

propRun1Types :: H.Property
propRun1Types = H.property $ do
  let execute = T.run1 T.CPU (T.transpose . T.map T.fromIntegral) :: T.TensorData '[2, 3] Int -> T.TensorData '[3, 2] Double
      first = T.run T.CPU (T.fromList @'[2, 3] [1 .. 6 :: Int])
      second = T.run T.CPU (T.fromList @'[2, 3] [7 .. 12 :: Int])
      total = T.run1 T.CPU T.sum :: T.TensorData '[2, 3] Int -> T.TensorData '[] Int
  T.toList (execute first) H.=== [1, 4, 2, 5, 3, 6]
  T.toList (execute second) H.=== [7, 10, 8, 11, 9, 12]
  T.toList (total first) H.=== [21]
  T.toList (total second) H.=== [57]

propRun2 :: H.Property
propRun2 = H.property $ do
  let execute = T.run2 T.CPU T.matmul :: T.TensorData '[2, 3] Int -> T.TensorData '[3, 2] Int -> T.TensorData '[2, 2] Int
      lhs = T.run T.CPU (T.fromList @'[2, 3] [1 .. 6 :: Int])
      rhs = T.run T.CPU (T.fromList @'[3, 2] [1 .. 6 :: Int])
      otherLeft = T.run T.CPU (T.ones @'[2, 3] @Int)
      otherRight = T.run T.CPU (T.full @'[3, 2] (2 :: Int))
      select = T.run2 T.CPU (\mask x -> T.select mask x 0) :: T.TensorData '[3] Bool -> T.TensorData '[3] Double -> T.TensorData '[3] Double
  T.toList (execute lhs rhs) H.=== [22, 28, 49, 64]
  T.toList (execute otherLeft rhs) H.=== [9, 12, 9, 12]
  T.toList (execute lhs otherRight) H.=== [12, 12, 30, 30]
  T.toList (execute lhs rhs) H.=== [22, 28, 49, 64]
  T.toList (select (T.run T.CPU (T.fromList @'[3] [True, False, True])) (T.run T.CPU (T.fromList @'[3] [1, 2, 3 :: Double]))) H.=== [1, 0, 3]

propPreparedEdges :: H.Property
propPreparedEdges = H.property $ do
  let scalar = T.run1 T.CPU (+ 1) :: T.TensorData '[] Int -> T.TensorData '[] Int
      empty = T.run1 T.CPU (T.broadcast @'[2, 0, 3]) :: T.TensorData '[1, 3] Int -> T.TensorData '[2, 0, 3] Int
      emptySum = T.run2 T.CPU (+) :: T.TensorData '[0] Int -> T.TensorData '[0] Int -> T.TensorData '[0] Int
      input = T.run T.CPU (T.zeros @'[0] @Int)
  T.toList (scalar (T.run T.CPU (T.scalar (T.constant (4 :: Int))))) H.=== [5]
  T.toList (empty (T.run T.CPU (T.ones @'[1, 3] @Int))) H.=== []
  T.toList (emptySum input input) H.=== []

propRun3 :: H.Property
propRun3 = H.property $ do
  let execute :: T.TensorData '[2, 2] Int -> T.TensorData '[2, 3] Int -> T.TensorData '[3] Int -> T.TensorData '[2, 3] Int
      execute = T.run3 T.CPU (\x w b -> T.matmul x w + T.broadcast b)
      inputs = T.run T.CPU (T.identity @2 @Int)
      weights = T.run T.CPU (T.fromList @'[2, 3] [1 .. 6 :: Int])
      bias = T.run T.CPU (T.fromList @'[3] [1, 2, 3 :: Int])
  T.toList (execute inputs weights bias) H.=== [2, 4, 6, 5, 7, 9]
  T.toList (execute (T.run T.CPU (T.ones @'[2, 2] @Int)) weights bias) H.=== [6, 9, 12, 6, 9, 12]
  T.toList (execute inputs (T.run T.CPU (T.ones @'[2, 3] @Int)) bias) H.=== [2, 3, 4, 2, 3, 4]
  T.toList (execute inputs weights (T.run T.CPU (T.zeros @'[3] @Int))) H.=== [1 .. 6]
  T.toList (execute inputs weights bias) H.=== [2, 4, 6, 5, 7, 9]

propRun3Types :: H.Property
propRun3Types = H.property $ do
  let execute :: T.TensorData '[2] Bool -> T.TensorData '[2] Int -> T.TensorData '[2] Double -> T.TensorData '[2] Double
      execute = T.run3 T.CPU (\m a b -> T.select m (T.map T.fromIntegral a + b) 0)
      mask = T.run T.CPU (T.fromList @'[2] [True, False])
      x = T.run T.CPU (T.fromList @'[2] [1, 2 :: Int])
      y = T.run T.CPU (T.fromList @'[2] [3, 4 :: Double])
  T.toList (execute mask x y) H.=== [4, 0]
  T.toList (execute (T.run T.CPU (T.not (T.fromData mask))) x y) H.=== [0, 6]

propRun4 :: H.Property
propRun4 = H.property $ do
  let execute :: T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int
      execute = T.run4 T.CPU (\a1 a2 a3 a4 -> a1 + 2 * a2 + 3 * a3 + 4 * a4)
      x1 = T.run T.CPU (T.fromList @'[2] [1, -1 :: Int])
      x2 = T.run T.CPU (T.fromList @'[2] [2, -2 :: Int])
      x3 = T.run T.CPU (T.fromList @'[2] [3, -3 :: Int])
      x4 = T.run T.CPU (T.fromList @'[2] [4, -4 :: Int])
  T.toList (execute x1 x2 x3 x4) H.=== [30, -30]
  T.toList (execute x4 x3 x2 x1) H.=== [20, -20]
  T.toList (execute x1 x2 x3 x4) H.=== [30, -30]

  let executeOutputs = T.run4 T.CPU (\firstTensor _ _ lastTensor -> (firstTensor, T.scalar (T.index @'[0] lastTensor)))
      (firstOutput, lastOutput) = executeOutputs x1 x2 x3 x4
  T.toList firstOutput H.=== [1, -1]
  T.toList lastOutput H.=== [4]

propRun5 :: H.Property
propRun5 = H.property $ do
  let execute :: T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int
      execute = T.run5 T.CPU (\a1 a2 a3 a4 a5 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5)
      x1 = T.run T.CPU (T.fromList @'[2] [1, -1 :: Int])
      x2 = T.run T.CPU (T.fromList @'[2] [2, -2 :: Int])
      x3 = T.run T.CPU (T.fromList @'[2] [3, -3 :: Int])
      x4 = T.run T.CPU (T.fromList @'[2] [4, -4 :: Int])
      x5 = T.run T.CPU (T.fromList @'[2] [5, -5 :: Int])
  T.toList (execute x1 x2 x3 x4 x5) H.=== [55, -55]
  T.toList (execute x5 x4 x3 x2 x1) H.=== [35, -35]
  T.toList (execute x1 x2 x3 x4 x5) H.=== [55, -55]

  let executeOutputs = T.run5 T.CPU (\firstTensor _ _ _ lastTensor -> (firstTensor, T.scalar (T.index @'[0] lastTensor)))
      (firstOutput, lastOutput) = executeOutputs x1 x2 x3 x4 x5
  T.toList firstOutput H.=== [1, -1]
  T.toList lastOutput H.=== [5]

propRun6 :: H.Property
propRun6 = H.property $ do
  let execute :: T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int
      execute = T.run6 T.CPU (\a1 a2 a3 a4 a5 a6 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6)
      x1 = T.run T.CPU (T.fromList @'[2] [1, -1 :: Int])
      x2 = T.run T.CPU (T.fromList @'[2] [2, -2 :: Int])
      x3 = T.run T.CPU (T.fromList @'[2] [3, -3 :: Int])
      x4 = T.run T.CPU (T.fromList @'[2] [4, -4 :: Int])
      x5 = T.run T.CPU (T.fromList @'[2] [5, -5 :: Int])
      x6 = T.run T.CPU (T.fromList @'[2] [6, -6 :: Int])
  T.toList (execute x1 x2 x3 x4 x5 x6) H.=== [91, -91]
  T.toList (execute x6 x5 x4 x3 x2 x1) H.=== [56, -56]
  T.toList (execute x1 x2 x3 x4 x5 x6) H.=== [91, -91]

  let executeOutputs = T.run6 T.CPU (\firstTensor _ _ _ _ lastTensor -> (firstTensor, T.scalar (T.index @'[0] lastTensor)))
      (firstOutput, lastOutput) = executeOutputs x1 x2 x3 x4 x5 x6
  T.toList firstOutput H.=== [1, -1]
  T.toList lastOutput H.=== [6]

propRun7 :: H.Property
propRun7 = H.property $ do
  let execute :: T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int
      execute = T.run7 T.CPU (\a1 a2 a3 a4 a5 a6 a7 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 + 7 * a7)
      x1 = T.run T.CPU (T.fromList @'[2] [1, -1 :: Int])
      x2 = T.run T.CPU (T.fromList @'[2] [2, -2 :: Int])
      x3 = T.run T.CPU (T.fromList @'[2] [3, -3 :: Int])
      x4 = T.run T.CPU (T.fromList @'[2] [4, -4 :: Int])
      x5 = T.run T.CPU (T.fromList @'[2] [5, -5 :: Int])
      x6 = T.run T.CPU (T.fromList @'[2] [6, -6 :: Int])
      x7 = T.run T.CPU (T.fromList @'[2] [7, -7 :: Int])
  T.toList (execute x1 x2 x3 x4 x5 x6 x7) H.=== [140, -140]
  T.toList (execute x7 x6 x5 x4 x3 x2 x1) H.=== [84, -84]
  T.toList (execute x1 x2 x3 x4 x5 x6 x7) H.=== [140, -140]

  let executeOutputs = T.run7 T.CPU (\firstTensor _ _ _ _ _ lastTensor -> (firstTensor, T.scalar (T.index @'[0] lastTensor)))
      (firstOutput, lastOutput) = executeOutputs x1 x2 x3 x4 x5 x6 x7
  T.toList firstOutput H.=== [1, -1]
  T.toList lastOutput H.=== [7]

propRun8 :: H.Property
propRun8 = H.property $ do
  let execute :: T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int
      execute = T.run8 T.CPU (\a1 a2 a3 a4 a5 a6 a7 a8 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 + 7 * a7 + 8 * a8)
      x1 = T.run T.CPU (T.fromList @'[2] [1, -1 :: Int])
      x2 = T.run T.CPU (T.fromList @'[2] [2, -2 :: Int])
      x3 = T.run T.CPU (T.fromList @'[2] [3, -3 :: Int])
      x4 = T.run T.CPU (T.fromList @'[2] [4, -4 :: Int])
      x5 = T.run T.CPU (T.fromList @'[2] [5, -5 :: Int])
      x6 = T.run T.CPU (T.fromList @'[2] [6, -6 :: Int])
      x7 = T.run T.CPU (T.fromList @'[2] [7, -7 :: Int])
      x8 = T.run T.CPU (T.fromList @'[2] [8, -8 :: Int])
  T.toList (execute x1 x2 x3 x4 x5 x6 x7 x8) H.=== [204, -204]
  T.toList (execute x8 x7 x6 x5 x4 x3 x2 x1) H.=== [120, -120]
  T.toList (execute x1 x2 x3 x4 x5 x6 x7 x8) H.=== [204, -204]

  let executeOutputs = T.run8 T.CPU (\firstTensor _ _ _ _ _ _ lastTensor -> (firstTensor, T.scalar (T.index @'[0] lastTensor)))
      (firstOutput, lastOutput) = executeOutputs x1 x2 x3 x4 x5 x6 x7 x8
  T.toList firstOutput H.=== [1, -1]
  T.toList lastOutput H.=== [8]

propRun9 :: H.Property
propRun9 = H.property $ do
  let execute :: T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int -> T.TensorData '[2] Int
      execute = T.run9 T.CPU (\a1 a2 a3 a4 a5 a6 a7 a8 a9 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 + 7 * a7 + 8 * a8 + 9 * a9)
      x1 = T.run T.CPU (T.fromList @'[2] [1, -1 :: Int])
      x2 = T.run T.CPU (T.fromList @'[2] [2, -2 :: Int])
      x3 = T.run T.CPU (T.fromList @'[2] [3, -3 :: Int])
      x4 = T.run T.CPU (T.fromList @'[2] [4, -4 :: Int])
      x5 = T.run T.CPU (T.fromList @'[2] [5, -5 :: Int])
      x6 = T.run T.CPU (T.fromList @'[2] [6, -6 :: Int])
      x7 = T.run T.CPU (T.fromList @'[2] [7, -7 :: Int])
      x8 = T.run T.CPU (T.fromList @'[2] [8, -8 :: Int])
      x9 = T.run T.CPU (T.fromList @'[2] [9, -9 :: Int])
      zipInputs = T.run9 T.CPU T.zip9
  T.toList (execute x1 x2 x3 x4 x5 x6 x7 x8 x9) H.=== [285, -285]
  T.toList (execute x9 x8 x7 x6 x5 x4 x3 x2 x1) H.=== [165, -165]
  T.toList (execute x1 x2 x3 x4 x5 x6 x7 x8 x9) H.=== [285, -285]
  T.toList (zipInputs x1 x2 x3 x4 x5 x6 x7 x8 x9) H.=== [(1, 2, 3, 4, 5, 6, 7, 8, 9), (-1, -2, -3, -4, -5, -6, -7, -8, -9)]

  let executeOutputs = T.run9 T.CPU (\firstTensor _ _ _ _ _ _ _ lastTensor -> (firstTensor, T.scalar (T.index @'[0] lastTensor)))
      (firstOutput, lastOutput) = executeOutputs x1 x2 x3 x4 x5 x6 x7 x8 x9
  T.toList firstOutput H.=== [1, -1]
  T.toList lastOutput H.=== [9]

propNestedComputation :: H.Property
propNestedComputation = H.property $ do
  let vector = T.fromList @'[3] [1, 2, 3 :: Int]
      matrix = T.fromList @'[2, 2] [1 .. 4 :: Double]
      result = (T.sum vector, (T.transpose matrix, vector T.> 1))
      (total, (transposed, mask)) = T.run T.CPU result
  T.toList total H.=== [6]
  T.toList transposed H.=== [1, 3, 2, 4]
  T.toList mask H.=== [False, True, True]

propRun1Outputs :: H.Property
propRun1Outputs = H.property $ do
  let execute :: T.TensorData '[2, 3] Int -> (T.TensorData '[] Int, T.TensorData '[3] Bool)
      execute = T.run1 T.CPU (\x -> (T.sum x, T.anyAxes @'[0] (x T.> 2)))
      first = T.run T.CPU (T.fromList @'[2, 3] [1 .. 6 :: Int])
      second = T.run T.CPU (T.fromList @'[2, 3] [0, 5, 0, 0, 0, 0 :: Int])
      (firstTotal, firstMask) = execute first
      (secondTotal, secondMask) = execute second
  T.toList firstTotal H.=== [21]
  T.toList firstMask H.=== [True, True, True]
  T.toList secondTotal H.=== [5]
  T.toList secondMask H.=== [False, True, False]

propRun2Outputs :: H.Property
propRun2Outputs = H.property $ do
  let execute :: T.TensorData '[2, 3] Int -> T.TensorData '[3, 2] Int -> (T.TensorData '[2, 2] Int, T.TensorData '[] Int)
      execute = T.run2 T.CPU (\x w -> let productTensor = T.matmul x w in (productTensor, T.sum productTensor))
      inputs = T.run T.CPU (T.fromList @'[2, 3] [1 .. 6 :: Int])
      weights = T.run T.CPU (T.fromList @'[3, 2] [1 .. 6 :: Int])
      (matrix, total) = execute inputs weights
  T.toList matrix H.=== [22, 28, 49, 64]
  T.toList total H.=== [163]

propRun3Outputs :: H.Property
propRun3Outputs = H.property $ do
  let execute :: T.TensorData '[2, 2] Int -> T.TensorData '[2, 3] Int -> T.TensorData '[3] Int -> (T.TensorData '[2, 3] Int, T.TensorData '[3] Int, T.TensorData '[2] Int)
      execute = T.run3 T.CPU (\x w b -> let scores = T.matmul x w + T.broadcast b in (scores, T.sumAxes @'[0] scores, T.argmax @1 scores))
      inputs = T.run T.CPU (T.identity @2 @Int)
      weights = T.run T.CPU (T.fromList @'[2, 3] [1 .. 6 :: Int])
      (logits, columns, classes) = execute inputs weights (T.run T.CPU (T.zeros @'[3] @Int))
      (newLogits, newColumns, newClasses) = execute inputs weights (T.run T.CPU (T.fromList @'[3] [10, 0, 0 :: Int]))
  T.toList logits H.=== [1 .. 6]
  T.toList columns H.=== [5, 7, 9]
  T.toList classes H.=== [2, 2]
  T.toList newLogits H.=== [11, 2, 3, 14, 5, 6]
  T.toList newColumns H.=== [25, 7, 9]
  T.toList newClasses H.=== [0, 0]

propComputationEdges :: H.Property
propComputationEdges = H.property $ do
  let (empty, scalar) = T.run T.CPU (T.zeros @'[2, 0, 3] @Int, T.scalar (T.constant True))
      execute :: T.TensorData '[0] Int -> (T.TensorData '[0] Bool, T.TensorData '[] Int)
      execute = T.run1 T.CPU (\x -> (x T.> 0, T.sum x))
      (emptyMask, total) = execute (T.run T.CPU (T.zeros @'[0] @Int))
  T.toList empty H.=== []
  T.toList scalar H.=== [True]
  T.toList emptyMask H.=== []
  T.toList total H.=== [0]

propOutputs2 :: H.Property
propOutputs2 = H.property $ do
  let x1 = T.full @'[1] (1 :: Int)
      x2 = T.full @'[2] (2 :: Int)
      (r1, r2) = T.run T.CPU (x1, x2)
  T.toList r1 H.=== replicate 1 1
  T.toList r2 H.=== replicate 2 2

propOutputs3 :: H.Property
propOutputs3 = H.property $ do
  let x1 = T.full @'[1] (1 :: Int)
      x2 = T.full @'[2] (2 :: Int)
      x3 = T.full @'[3] (3 :: Int)
      (r1, r2, r3) = T.run T.CPU (x1, x2, x3)
  T.toList r1 H.=== replicate 1 1
  T.toList r2 H.=== replicate 2 2
  T.toList r3 H.=== replicate 3 3

propOutputs4 :: H.Property
propOutputs4 = H.property $ do
  let x1 = T.full @'[1] (1 :: Int)
      x2 = T.full @'[2] (2 :: Int)
      x3 = T.full @'[3] (3 :: Int)
      x4 = T.full @'[4] (4 :: Int)
      (r1, r2, r3, r4) = T.run T.CPU (x1, x2, x3, x4)
  T.toList r1 H.=== replicate 1 1
  T.toList r2 H.=== replicate 2 2
  T.toList r3 H.=== replicate 3 3
  T.toList r4 H.=== replicate 4 4

propOutputs5 :: H.Property
propOutputs5 = H.property $ do
  let x1 = T.full @'[1] (1 :: Int)
      x2 = T.full @'[2] (2 :: Int)
      x3 = T.full @'[3] (3 :: Int)
      x4 = T.full @'[4] (4 :: Int)
      x5 = T.full @'[5] (5 :: Int)
      (r1, r2, r3, r4, r5) = T.run T.CPU (x1, x2, x3, x4, x5)
  T.toList r1 H.=== replicate 1 1
  T.toList r2 H.=== replicate 2 2
  T.toList r3 H.=== replicate 3 3
  T.toList r4 H.=== replicate 4 4
  T.toList r5 H.=== replicate 5 5

propOutputs6 :: H.Property
propOutputs6 = H.property $ do
  let x1 = T.full @'[1] (1 :: Int)
      x2 = T.full @'[2] (2 :: Int)
      x3 = T.full @'[3] (3 :: Int)
      x4 = T.full @'[4] (4 :: Int)
      x5 = T.full @'[5] (5 :: Int)
      x6 = T.full @'[6] (6 :: Int)
      (r1, r2, r3, r4, r5, r6) = T.run T.CPU (x1, x2, x3, x4, x5, x6)
  T.toList r1 H.=== replicate 1 1
  T.toList r2 H.=== replicate 2 2
  T.toList r3 H.=== replicate 3 3
  T.toList r4 H.=== replicate 4 4
  T.toList r5 H.=== replicate 5 5
  T.toList r6 H.=== replicate 6 6

propOutputs7 :: H.Property
propOutputs7 = H.property $ do
  let x1 = T.full @'[1] (1 :: Int)
      x2 = T.full @'[2] (2 :: Int)
      x3 = T.full @'[3] (3 :: Int)
      x4 = T.full @'[4] (4 :: Int)
      x5 = T.full @'[5] (5 :: Int)
      x6 = T.full @'[6] (6 :: Int)
      x7 = T.full @'[7] (7 :: Int)
      (r1, r2, r3, r4, r5, r6, r7) = T.run T.CPU (x1, x2, x3, x4, x5, x6, x7)
  T.toList r1 H.=== replicate 1 1
  T.toList r2 H.=== replicate 2 2
  T.toList r3 H.=== replicate 3 3
  T.toList r4 H.=== replicate 4 4
  T.toList r5 H.=== replicate 5 5
  T.toList r6 H.=== replicate 6 6
  T.toList r7 H.=== replicate 7 7

propOutputs8 :: H.Property
propOutputs8 = H.property $ do
  let x1 = T.full @'[1] (1 :: Int)
      x2 = T.full @'[2] (2 :: Int)
      x3 = T.full @'[3] (3 :: Int)
      x4 = T.full @'[4] (4 :: Int)
      x5 = T.full @'[5] (5 :: Int)
      x6 = T.full @'[6] (6 :: Int)
      x7 = T.full @'[7] (7 :: Int)
      x8 = T.full @'[8] (8 :: Int)
      (r1, r2, r3, r4, r5, r6, r7, r8) = T.run T.CPU (x1, x2, x3, x4, x5, x6, x7, x8)
  T.toList r1 H.=== replicate 1 1
  T.toList r2 H.=== replicate 2 2
  T.toList r3 H.=== replicate 3 3
  T.toList r4 H.=== replicate 4 4
  T.toList r5 H.=== replicate 5 5
  T.toList r6 H.=== replicate 6 6
  T.toList r7 H.=== replicate 7 7
  T.toList r8 H.=== replicate 8 8

propOutputs9 :: H.Property
propOutputs9 = H.property $ do
  let x1 = T.full @'[1] (1 :: Int)
      x2 = T.full @'[2] (2 :: Int)
      x3 = T.full @'[3] (3 :: Int)
      x4 = T.full @'[4] (4 :: Int)
      x5 = T.full @'[5] (5 :: Int)
      x6 = T.full @'[6] (6 :: Int)
      x7 = T.full @'[7] (7 :: Int)
      x8 = T.full @'[8] (8 :: Int)
      x9 = T.full @'[9] (9 :: Int)
      (r1, r2, r3, r4, r5, r6, r7, r8, r9) = T.run T.CPU (x1, x2, x3, x4, x5, x6, x7, x8, x9)
  T.toList r1 H.=== replicate 1 1
  T.toList r2 H.=== replicate 2 2
  T.toList r3 H.=== replicate 3 3
  T.toList r4 H.=== replicate 4 4
  T.toList r5 H.=== replicate 5 5
  T.toList r6 H.=== replicate 6 6
  T.toList r7 H.=== replicate 7 7
  T.toList r8 H.=== replicate 8 8
  T.toList r9 H.=== replicate 9 9
