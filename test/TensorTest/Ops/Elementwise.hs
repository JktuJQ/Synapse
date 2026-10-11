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
    [ testProperty "map composition agrees with a single map" propMapComposition,
      testProperty "zipWith agrees with list combination" propZipWithLists,
      testProperty "map can change element type" propMap,
      testProperty "zip pairs different element types" (H.withTests 1 propZip),
      testProperty "zipWith supports distinct input and output types" (H.withTests 1 propZipWith),
      testProperty "element operations handle empty tensors and scalars" (H.withTests 1 propElementOperationEdges),
      testProperty "boolean operators agree with truth tables" (H.withTests 1 propBoolean),
      testProperty "tensor masks select corresponding values" propSelect,
      testProperty "element conditions compose inside map" (H.withTests 1 propElementSelect),
      testProperty "scalar conditions select whole tensor computations" (H.withTests 1 propCond),
      testProperty "selection supports scalars, empty tensors, and tuples" (H.withTests 1 propSelectEdges),
      testProperty "while checks its predicate before each update" (H.withTests 1 propWhile),
      testProperty "pair states support multiple loop variables" (H.withTests 1 propWhilePairs),
      testProperty "iteration counts may depend on generator indices" (H.withTests 1 propIterate),
      testProperty "integer quotients truncate toward zero" propIntegral,
      testProperty "floating classification distinguishes NaN and infinities" (H.withTests 1 propClassification),
      testProperty "valid conditional operations compile" (H.withTests 1 (Compiler.accepts (root ++ "Accepted/Control.hs"))),
      testProperty "select mask shape must match" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/SelectMask.hs") "Couldn't match type")),
      testProperty "select branch shapes must match" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/SelectShape.hs") "Couldn't match type")),
      testProperty "cond branch shapes must match" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/CondShape.hs") "Couldn't match type")),
      testProperty "boolean operand shapes must match" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/BooleanShape.hs") "Couldn't match type")),
      testProperty "while must preserve its state type" (H.withTests 1 (Compiler.rejects (root ++ "Rejected/WhileState.hs") "Couldn't match expected type"))
    ]
  where
    root = "test/TensorTest/Ops/Elementwise/Fixtures/"

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
              let converted = T.fromIntegral x :: T.TensorElement T.Float32
               in T.select flag (converted + 0.5) (converted - 0.5)
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
  T.toList (T.runCPU (T.map (+ 1) (T.scalar (T.constant (2 :: Int))))) H.=== [3]
  T.toList (T.runCPU (T.zip (T.scalar (T.constant (2 :: Int))) (T.scalar (T.constant True)))) H.=== [(2, True)]

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

propBoolean :: H.Property
propBoolean = H.property $ do
  let xs = [False, False, True, True]
      ys = [False, True, False, True]
      lhs = T.fromList @'[4] xs
      rhs = T.fromList @'[4] ys
  T.toList (T.runCPU (T.not lhs)) H.=== map not xs
  T.toList (T.runCPU (lhs T.&& rhs)) H.=== zipWith (&&) xs ys
  T.toList (T.runCPU (lhs T.|| rhs)) H.=== zipWith (||) xs ys
  T.toList (T.runCPU (T.zipWith (T.&&) lhs rhs)) H.=== zipWith (&&) xs ys
  T.toList (T.runCPU (T.map T.not lhs)) H.=== map not xs

propSelect :: H.Property
propSelect = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  flags <- H.forAll $ Gen.list (Range.singleton 6) Gen.bool
  let lhs = T.fromList @'[2, 3] xs
      rhs = T.fromList @'[2, 3] ys
      mask = T.fromList @'[2, 3] flags
  T.toList (T.runCPU (T.select mask lhs rhs)) H.=== zipWith3 (\p x y -> if p then x else y) flags xs ys
  T.toList (T.runCPU (T.select ((lhs T.> 0) T.&& (lhs T.< 50)) lhs 0)) H.=== map (\x -> if x > 0 && x < 50 then x else 0) xs

propElementSelect :: H.Property
propElementSelect = H.property $ do
  let input = T.fromList @'[3] [-2, 0, 3 :: Int]
  T.toList (T.runCPU (T.map (\x -> T.select (x T.< 0) (negate x) x) input)) H.=== [2, 0, 3]
  T.toList (T.runCPU (T.map (\x -> T.cond (x T.> 0) (x * 2) x) input)) H.=== [-2, 0, 6]

propCond :: H.Property
propCond = H.property $ do
  let lhs = T.fromList @'[3] [1, 2, 3 :: Int]
      rhs = T.fromList @'[3] [4, 5, 6 :: Int]
  T.toList (T.runCPU (T.cond (T.sum lhs T.> 0) (lhs + 1) rhs)) H.=== [2, 3, 4]
  T.toList (T.runCPU (T.cond (T.constant False) lhs rhs)) H.=== [4, 5, 6]

propSelectEdges :: H.Property
propSelectEdges = H.property $ do
  let empty = T.zeros @'[2, 0] :: T.Tensor '[2, 0] Int
      emptyMask = T.full @'[2, 0] True
      scalarMask = T.scalar (T.constant False)
  T.toList (T.runCPU (T.select emptyMask empty empty)) H.=== []
  T.toList (T.runCPU (T.not emptyMask T.|| emptyMask)) H.=== []
  T.toList (T.runCPU (T.select scalarMask (T.scalar (T.constant True)) (T.scalar (T.constant False)))) H.=== [False]
  let chosen = T.select (T.constant True) (T.pair (T.constant (3 :: Int)) (T.constant True)) (T.pair 7 (T.constant False))
  T.toList (T.runCPU (T.scalar chosen)) H.=== [(3, True)]
  T.toList (T.runCPU (T.scalar (T.fst chosen))) H.=== [3]
  T.toList (T.runCPU (T.scalar (T.snd chosen))) H.=== [True]
  let lhs = T.fromList @'[2] [(1 :: Int, True), (2, False)]
      rhs = T.fromList @'[2] [(3 :: Int, False), (4, True)]
  T.toList (T.runCPU (T.select (T.fromList @'[2] [True, False]) lhs rhs)) H.=== [(1, True), (4, True)]

propWhile :: H.Property
propWhile = H.property $ do
  T.toList (T.runCPU (T.scalar (T.while (T.< 10) (+ 2) (1 :: T.TensorElement Int)))) H.=== [11]
  T.toList (T.runCPU (T.scalar (T.while (T.< 0) (+ 1) (3 :: T.TensorElement Int)))) H.=== [3]
  let input = T.fromList @'[3] [1, 3, 5 :: Int]
  T.toList (T.runCPU (T.map (T.while (T.< 5) (+ 1)) input)) H.=== [5, 5, 5]

propWhilePairs :: H.Property
propWhilePairs = H.property $ do
  let factorial :: T.TensorElement Int -> T.TensorElement Int
      factorial limit =
        T.snd $
          T.while
            (\state -> T.fst state T.<= limit)
            (\state -> T.pair (T.fst state + 1) (T.snd state * T.fst state))
            (T.pair 1 1)
      result = T.fromGen @'[5] (factorial . T.coordinate @0)
  T.toList (T.runCPU result) H.=== [1, 1, 2, 6, 24 :: Int]

propIterate :: H.Property
propIterate = H.property $ do
  let result = T.fromGen @'[4] (\ix -> T.iterate (T.coordinate @0 ix) (* 2) (1 :: T.TensorElement Int))
  T.toList (T.runCPU result) H.=== [1, 2, 4, 8]
  T.toList (T.runCPU (T.scalar (T.iterate 0 (+ 1) (7 :: T.TensorElement Int)))) H.=== [7]

propIntegral :: H.Property
propIntegral = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 4) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 4) (Gen.element [-7, -3, 2, 5])
  let lhs = T.fromList @'[4] xs
      rhs = T.fromList @'[4] ys
  T.toList (T.runCPU (T.quot lhs rhs)) H.=== zipWith quot xs ys
  T.toList (T.runCPU (T.rem lhs rhs)) H.=== zipWith rem xs ys
  T.toList (T.runCPU (T.map (`T.quot` 3) lhs)) H.=== map (`quot` 3) xs
  T.toList (T.runCPU (T.map (`T.rem` (-3)) lhs)) H.=== map (`rem` (-3)) xs

propClassification :: H.Property
propClassification = H.property $ do
  let input = T.fromList @'[5] [0 / 0, 1 / 0, -(1 / 0), 0, -3 :: Double]
      single = T.fromList @'[3] [0 / 0, 1 / 0, 2 :: Float]
  T.toList (T.runCPU (T.isNaN input)) H.=== [True, False, False, False, False]
  T.toList (T.runCPU (T.isInfinite input)) H.=== [False, True, True, False, False]
  T.toList (T.runCPU (T.map T.isNaN single)) H.=== [True, False, False]
  T.toList (T.runCPU (T.map T.isInfinite single)) H.=== [False, True, False]
  T.toList (T.runCPU (T.isNaN (T.zeros @'[0] :: T.Tensor '[0] Double))) H.=== []
