{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Elementwise (tests) where

import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "elementwise"
    [ testProperty "boolean operators agree with truth tables" (H.withTests 1 propBoolean),
      testProperty "tensor masks select corresponding values" propSelect,
      testProperty "element conditions compose inside generators" (H.withTests 1 propElementSelect),
      testProperty "scalar conditions select whole tensor computations" (H.withTests 1 propCond),
      testProperty "selection supports scalars, empty tensors, and tuples" (H.withTests 1 propSelectEdges),
      testProperty "integer quotients truncate toward zero" propIntegral,
      testProperty "floating classification distinguishes NaN and infinities" (H.withTests 1 propClassification),
      testProperty "valid conditional operations compile" (H.withTests 1 (Compiler.accepts (root ++ "Accepted/Control.hs"))),
      testProperty "select mask shape must match" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/SelectMask.hs") "Couldn't match type")),
      testProperty "select branch shapes must match" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/SelectShape.hs") "Couldn't match type")),
      testProperty "cond branch shapes must match" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/CondShape.hs") "Couldn't match type")),
      testProperty "boolean operand shapes must match" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/BooleanShape.hs") "Couldn't match type"))
    ]
  where
    root = "test/TensorTest/Ops/Elementwise/Fixtures/"

propBoolean :: H.Property
propBoolean = H.property $ do
  let xs = [False, False, True, True]
      ys = [False, True, False, True]
      lhs = T.fromList @'[4] xs
      rhs = T.fromList @'[4] ys
  T.toList (T.run T.CPU (T.not lhs)) H.=== map not xs
  T.toList (T.run T.CPU (lhs T.&& rhs)) H.=== zipWith (&&) xs ys
  T.toList (T.run T.CPU (lhs T.|| rhs)) H.=== zipWith (||) xs ys

propSelect :: H.Property
propSelect = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  flags <- H.forAll $ Gen.list (Range.singleton 6) Gen.bool
  let lhs = T.fromList @'[2, 3] xs
      rhs = T.fromList @'[2, 3] ys
      mask = T.fromList @'[2, 3] flags
  T.toList (T.run T.CPU (T.select mask lhs rhs)) H.=== zipWith3 (\p x y -> if p then x else y) flags xs ys
  T.toList (T.run T.CPU (T.select ((lhs T.> 0) T.&& (lhs T.< 50)) lhs 0)) H.=== map (\x -> if x > 0 && x < 50 then x else 0) xs

propElementSelect :: H.Property
propElementSelect = H.property $ do
  let input = T.fromList @'[3] [-2, 0, 3 :: Int]
  T.toList (T.run T.CPU (T.fromGen @'[3] (\ix -> let x = input T.!! ix in T.select (x T.< 0) (negate x) x))) H.=== [2, 0, 3]
  T.toList (T.run T.CPU (T.fromGen @'[3] (\ix -> let x = input T.!! ix in T.cond (x T.> 0) (x * 2) x))) H.=== [-2, 0, 6]

propCond :: H.Property
propCond = H.property $ do
  let lhs = T.fromList @'[3] [1, 2, 3 :: Int]
      rhs = T.fromList @'[3] [4, 5, 6 :: Int]
  T.toList (T.run T.CPU (T.cond (T.sum lhs T.> 0) (lhs + 1) rhs)) H.=== [2, 3, 4]
  T.toList (T.run T.CPU (T.cond (T.scalar (T.constant False)) lhs rhs)) H.=== [4, 5, 6]

propSelectEdges :: H.Property
propSelectEdges = H.property $ do
  let empty = T.zeros @'[2, 0] :: T.Tensor '[2, 0] Int
      emptyMask = T.full @'[2, 0] True
      scalarMask = T.scalar (T.constant False)
  T.toList (T.run T.CPU (T.select emptyMask empty empty)) H.=== []
  T.toList (T.run T.CPU (T.not emptyMask T.|| emptyMask)) H.=== []
  T.toList (T.run T.CPU (T.select scalarMask (T.scalar (T.constant True)) (T.scalar (T.constant False)))) H.=== [False]
  let lhs = T.fromList @'[2] [(1 :: Int, True), (2, False)]
      rhs = T.fromList @'[2] [(3 :: Int, False), (4, True)]
  T.toList (T.run T.CPU (T.select (T.fromList @'[2] [True, False]) lhs rhs)) H.=== [(1, True), (4, True)]

propIntegral :: H.Property
propIntegral = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 4) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 4) (Gen.element [-7, -3, 2, 5])
  let lhs = T.fromList @'[4] xs
      rhs = T.fromList @'[4] ys
  T.toList (T.run T.CPU (T.quot lhs rhs)) H.=== zipWith quot xs ys
  T.toList (T.run T.CPU (T.rem lhs rhs)) H.=== zipWith rem xs ys
  T.toList (T.run T.CPU (T.quot lhs 3)) H.=== map (`quot` 3) xs
  T.toList (T.run T.CPU (T.rem lhs (-3))) H.=== map (`rem` (-3)) xs

propClassification :: H.Property
propClassification = H.property $ do
  let input = T.fromList @'[5] [0 / 0, 1 / 0, -(1 / 0), 0, -3 :: Double]
      single = T.fromList @'[3] [0 / 0, 1 / 0, 2 :: Float]
  T.toList (T.run T.CPU (T.isNaN input)) H.=== [True, False, False, False, False]
  T.toList (T.run T.CPU (T.isInfinite input)) H.=== [False, True, True, False, False]
  T.toList (T.run T.CPU (T.isNaN single)) H.=== [True, False, False]
  T.toList (T.run T.CPU (T.isInfinite single)) H.=== [False, True, False]
  T.toList (T.run T.CPU (T.isNaN (T.zeros @'[0] :: T.Tensor '[0] Double))) H.=== []
