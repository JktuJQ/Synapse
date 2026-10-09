{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Math (tests) where

import Control.Monad (forM_)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "math"
    [ testProperty "integer arithmetic agrees with lists" propIntegerArithmetic,
      testProperty "Floating methods agree with host calculations" (H.withTests 1 propFloatingMethods),
      testProperty "Num instance is elementwise" (H.withTests 1 propNumInstance),
      testProperty "Fractional instance is elementwise" (H.withTests 1 propFractionalInstance),
      testProperty "Floating instance is elementwise" (H.withTests 1 propFloatingInstance)
    ]

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

propIntegerArithmetic :: H.Property
propIntegerArithmetic = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let lhs = T.fromList @'[2, 3] xs
      rhs = T.fromList @'[2, 3] ys
  T.toList (T.runCPU (lhs + rhs)) H.=== zipWith (+) xs ys
  T.toList (T.runCPU (lhs - rhs)) H.=== zipWith (-) xs ys
  T.toList (T.runCPU (lhs * rhs)) H.=== zipWith (*) xs ys
  T.toList (T.runCPU (abs lhs)) H.=== map abs xs
  T.toList (T.runCPU (signum lhs)) H.=== map signum xs
  T.toList (T.runCPU (negate lhs)) H.=== map negate xs

propFloatingMethods :: H.Property
propFloatingMethods = H.property $ do
  let xs = [0.2, 0.3, 0.5, 0.7 :: T.Float64]
      input = T.fromList @'[4] xs
      operations :: [(String, T.Tensor '[4] T.Float64 -> T.Tensor '[4] T.Float64, T.Float64 -> T.Float64)]
      operations =
        [ ("exp", exp, exp),
          ("log", log, log),
          ("sqrt", sqrt, sqrt),
          ("sin", sin, sin),
          ("cos", cos, cos),
          ("tan", tan, tan),
          ("asin", asin, asin),
          ("acos", acos, acos),
          ("atan", atan, atan),
          ("sinh", sinh, sinh),
          ("cosh", cosh, cosh),
          ("tanh", tanh, tanh),
          ("asinh", asinh, asinh),
          ("acosh", acosh . (+ 1), acosh . (+ 1)),
          ("atanh", atanh, atanh),
          ("power", (** 2.5), (** 2.5)),
          ("logBase", logBase 2, logBase 2)
        ]
  forM_ operations $ \(name, tensorOp, hostOp) -> do
    H.annotate name
    approxList (T.toList (T.runCPU (tensorOp input))) (map hostOp xs)
