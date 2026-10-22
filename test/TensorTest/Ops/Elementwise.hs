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
      testProperty "zip3 groups three distinct element types" (H.withTests 1 propZip3),
      testProperty "zipWith3 agrees with list combination" propZipWith3,
      testProperty "triple element operations support scalars and empty tensors" (H.withTests 1 propTripleEdges),
      testProperty "zip4 and zipWith4 preserve all input positions" (H.withTests 1 propZip4),
      testProperty "zip5 and zipWith5 preserve all input positions" (H.withTests 1 propZip5),
      testProperty "zip6 and zipWith6 preserve all input positions" (H.withTests 1 propZip6),
      testProperty "zip7 and zipWith7 preserve all input positions" (H.withTests 1 propZip7),
      testProperty "zip8 and zipWith8 preserve all input positions" (H.withTests 1 propZip8),
      testProperty "zip9 and zipWith9 preserve all input positions" (H.withTests 1 propZip9),
      testProperty "nine-input element operations support mixed types, scalars, and empty tensors" (H.withTests 1 propNineEdges),
      testProperty "unzip preserves component order and zip round-trips" (H.withTests 1 propUnzip2),
      testProperty "unzip3 preserves component order and zip round-trips" (H.withTests 1 propUnzip3),
      testProperty "unzip4 preserves component order and zip round-trips" (H.withTests 1 propUnzip4),
      testProperty "unzip5 preserves component order and zip round-trips" (H.withTests 1 propUnzip5),
      testProperty "unzip6 preserves component order and zip round-trips" (H.withTests 1 propUnzip6),
      testProperty "unzip7 preserves component order and zip round-trips" (H.withTests 1 propUnzip7),
      testProperty "unzip8 preserves component order and zip round-trips" (H.withTests 1 propUnzip8),
      testProperty "unzip9 preserves component order and zip round-trips" (H.withTests 1 propUnzip9),
      testProperty "unzip supports mixed and nested element types" (H.withTests 1 propUnzipTypes),
      testProperty "unzip preserves scalars and empty shapes" (H.withTests 1 propUnzipEdges),
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
  T.toList (T.run T.CPU output) H.=== Prelude.map (\x -> fromIntegral x + 0.5) xs

propZip :: H.Property
propZip = H.property $ do
  let lhs = T.fromList @'[3] [1, 2, 3 :: Int]
      rhs = T.fromList @'[3] [True, False, True]
      paired = T.zip lhs rhs
      nested = T.zip paired lhs
  T.toList (T.run T.CPU paired) H.=== [(1, True), (2, False), (3, True)]
  T.toList (T.run T.CPU nested) H.=== [((1, True), 1), ((2, False), 2), ((3, True), 3)]

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
  T.toList (T.run T.CPU output) H.=== [1.5, 1.5, 3.5 :: T.Float32]

propElementOperationEdges :: H.Property
propElementOperationEdges = H.property $ do
  let empty = T.zeros @'[2, 0] :: T.Tensor '[2, 0] Int
      mapped = T.map (\_ -> T.constant True) empty
  T.toList (T.run T.CPU mapped) H.=== []
  T.toList (T.run T.CPU (T.zip empty mapped)) H.=== []
  T.toList (T.run T.CPU (T.zipWith (\x _ -> x + 1) empty mapped)) H.=== []
  T.toList (T.run T.CPU (T.map (+ 1) (T.scalar (T.constant (2 :: Int))))) H.=== [3]
  T.toList (T.run T.CPU (T.zip (T.scalar (T.constant (2 :: Int))) (T.scalar (T.constant True)))) H.=== [(2, True)]

propMapComposition :: H.Property
propMapComposition = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3] xs
      composed = T.map (+ 3) (T.map (* 2) source)
      single = T.map (\x -> x * 2 + 3) source
  T.toList (T.run T.CPU composed) H.=== map (\x -> x * 2 + 3) xs
  T.toList (T.run T.CPU single) H.=== T.toList (T.run T.CPU composed)

propZipWithLists :: H.Property
propZipWithLists = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  let lhs = T.fromList @'[2, 3] xs
      rhs = T.fromList @'[2, 3] ys
      combined = T.zipWith (\x y -> x * 2 - y) lhs rhs
  T.toList (T.run T.CPU combined) H.=== zipWith (\x y -> x * 2 - y) xs ys
  T.toList (T.run T.CPU (T.zip lhs rhs)) H.=== zip xs ys

propBoolean :: H.Property
propBoolean = H.property $ do
  let xs = [False, False, True, True]
      ys = [False, True, False, True]
      lhs = T.fromList @'[4] xs
      rhs = T.fromList @'[4] ys
  T.toList (T.run T.CPU (T.not lhs)) H.=== map not xs
  T.toList (T.run T.CPU (lhs T.&& rhs)) H.=== zipWith (&&) xs ys
  T.toList (T.run T.CPU (lhs T.|| rhs)) H.=== zipWith (||) xs ys
  T.toList (T.run T.CPU (T.zipWith (T.&&) lhs rhs)) H.=== zipWith (&&) xs ys
  T.toList (T.run T.CPU (T.map T.not lhs)) H.=== map not xs

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
  T.toList (T.run T.CPU (T.map (\x -> T.select (x T.< 0) (negate x) x) input)) H.=== [2, 0, 3]
  T.toList (T.run T.CPU (T.map (\x -> T.cond (x T.> 0) (x * 2) x) input)) H.=== [-2, 0, 6]

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
  let chosen = T.select (T.constant True) (T.pair (T.constant (3 :: Int)) (T.constant True)) (T.pair 7 (T.constant False))
  T.toList (T.run T.CPU (T.scalar chosen)) H.=== [(3, True)]
  T.toList (T.run T.CPU (T.scalar (T.fst chosen))) H.=== [3]
  T.toList (T.run T.CPU (T.scalar (T.snd chosen))) H.=== [True]
  let lhs = T.fromList @'[2] [(1 :: Int, True), (2, False)]
      rhs = T.fromList @'[2] [(3 :: Int, False), (4, True)]
  T.toList (T.run T.CPU (T.select (T.fromList @'[2] [True, False]) lhs rhs)) H.=== [(1, True), (4, True)]

propWhile :: H.Property
propWhile = H.property $ do
  T.toList (T.run T.CPU (T.scalar (T.while (T.< 10) (+ 2) (1 :: T.TensorElement Int)))) H.=== [11]
  T.toList (T.run T.CPU (T.scalar (T.while (T.< 0) (+ 1) (3 :: T.TensorElement Int)))) H.=== [3]
  let input = T.fromList @'[3] [1, 3, 5 :: Int]
  T.toList (T.run T.CPU (T.map (T.while (T.< 5) (+ 1)) input)) H.=== [5, 5, 5]

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
  T.toList (T.run T.CPU result) H.=== [1, 1, 2, 6, 24 :: Int]

propIterate :: H.Property
propIterate = H.property $ do
  let result = T.fromGen @'[4] (\ix -> T.iterate (T.coordinate @0 ix) (* 2) (1 :: T.TensorElement Int))
  T.toList (T.run T.CPU result) H.=== [1, 2, 4, 8]
  T.toList (T.run T.CPU (T.scalar (T.iterate 0 (+ 1) (7 :: T.TensorElement Int)))) H.=== [7]

propIntegral :: H.Property
propIntegral = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 4) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 4) (Gen.element [-7, -3, 2, 5])
  let lhs = T.fromList @'[4] xs
      rhs = T.fromList @'[4] ys
  T.toList (T.run T.CPU (T.quot lhs rhs)) H.=== zipWith quot xs ys
  T.toList (T.run T.CPU (T.rem lhs rhs)) H.=== zipWith rem xs ys
  T.toList (T.run T.CPU (T.map (`T.quot` 3) lhs)) H.=== map (`quot` 3) xs
  T.toList (T.run T.CPU (T.map (`T.rem` (-3)) lhs)) H.=== map (`rem` (-3)) xs

propClassification :: H.Property
propClassification = H.property $ do
  let input = T.fromList @'[5] [0 / 0, 1 / 0, -(1 / 0), 0, -3 :: Double]
      single = T.fromList @'[3] [0 / 0, 1 / 0, 2 :: Float]
  T.toList (T.run T.CPU (T.isNaN input)) H.=== [True, False, False, False, False]
  T.toList (T.run T.CPU (T.isInfinite input)) H.=== [False, True, True, False, False]
  T.toList (T.run T.CPU (T.map T.isNaN single)) H.=== [True, False, False]
  T.toList (T.run T.CPU (T.map T.isInfinite single)) H.=== [False, True, False]
  T.toList (T.run T.CPU (T.isNaN (T.zeros @'[0] :: T.Tensor '[0] Double))) H.=== []

propZip3 :: H.Property
propZip3 = H.property $ do
  let integers = T.fromList @'[3] [1, 2, 3 :: Int]
      flags = T.fromList @'[3] [True, False, True]
      floats = T.fromList @'[3] [0.5, 1.5, 2.5 :: Double]
      result = T.zip3 integers flags floats
      materialized = T.run T.CPU result
  T.toList materialized H.=== [(1, True, 0.5), (2, False, 1.5), (3, True, 2.5)]
  T.toList (T.run T.CPU (T.fromData materialized)) H.=== T.toList materialized
  T.toList (T.run T.CPU (T.zip result integers)) H.=== [((1, True, 0.5), 1), ((2, False, 1.5), 2), ((3, True, 2.5), 3)]

propZipWith3 :: H.Property
propZipWith3 = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-100) 100))
  ys <- H.forAll $ Gen.list (Range.singleton 6) (Gen.element [0.5, 1.5, 2.5 :: Double])
  flags <- H.forAll $ Gen.list (Range.singleton 6) Gen.bool
  let integers = T.fromList @'[2, 3] xs
      floats = T.fromList @'[2, 3] ys
      mask = T.fromList @'[2, 3] flags
      result = T.zipWith3 (\x y flag -> T.select flag (T.fromIntegral x + y) 0) integers floats mask
      expected = zipWith3 (\x y flag -> if flag then fromIntegral x + y else 0) xs ys flags
  T.toList (T.run T.CPU result) H.=== expected

propTripleEdges :: H.Property
propTripleEdges = H.property $ do
  let emptyInt = T.zeros @'[2, 0] @Int
      emptyBool = T.full @'[2, 0] False
      emptyFloat = T.zeros @'[2, 0] @Double
      scalarInt = T.scalar (T.constant (3 :: Int))
      scalarBool = T.scalar (T.constant True)
      scalarFloat = T.scalar (T.constant (0.5 :: Double))
  T.toList (T.run T.CPU (T.zip3 emptyInt emptyBool emptyFloat)) H.=== []
  T.toList (T.run T.CPU (T.zipWith3 (\x flag y -> T.select flag (T.fromIntegral x + y) 0) emptyInt emptyBool emptyFloat)) H.=== []
  T.toList (T.run T.CPU (T.zip3 scalarInt scalarBool scalarFloat)) H.=== [(3, True, 0.5)]
  T.toList (T.run T.CPU (T.zipWith3 (\x flag y -> T.select flag (T.fromIntegral x + y) 0) scalarInt scalarBool scalarFloat)) H.=== [3.5]

propZip4 :: H.Property
propZip4 = H.property $ do
  let x1 = T.fromList @'[2] [1, -1 :: Int]
      x2 = T.fromList @'[2] [2, -2 :: Int]
      x3 = T.fromList @'[2] [3, -3 :: Int]
      x4 = T.fromList @'[2] [4, -4 :: Int]
  T.toList (T.run T.CPU (T.zip4 x1 x2 x3 x4)) H.=== [(1, 2, 3, 4), (-1, -2, -3, -4)]
  T.toList (T.run T.CPU (T.zipWith4 (\a1 a2 a3 a4 -> a1 + 2 * a2 + 3 * a3 + 4 * a4) x1 x2 x3 x4)) H.=== [30, -30]

propZip5 :: H.Property
propZip5 = H.property $ do
  let x1 = T.fromList @'[2] [1, -1 :: Int]
      x2 = T.fromList @'[2] [2, -2 :: Int]
      x3 = T.fromList @'[2] [3, -3 :: Int]
      x4 = T.fromList @'[2] [4, -4 :: Int]
      x5 = T.fromList @'[2] [5, -5 :: Int]
  T.toList (T.run T.CPU (T.zip5 x1 x2 x3 x4 x5)) H.=== [(1, 2, 3, 4, 5), (-1, -2, -3, -4, -5)]
  T.toList (T.run T.CPU (T.zipWith5 (\a1 a2 a3 a4 a5 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5) x1 x2 x3 x4 x5)) H.=== [55, -55]

propZip6 :: H.Property
propZip6 = H.property $ do
  let x1 = T.fromList @'[2] [1, -1 :: Int]
      x2 = T.fromList @'[2] [2, -2 :: Int]
      x3 = T.fromList @'[2] [3, -3 :: Int]
      x4 = T.fromList @'[2] [4, -4 :: Int]
      x5 = T.fromList @'[2] [5, -5 :: Int]
      x6 = T.fromList @'[2] [6, -6 :: Int]
  T.toList (T.run T.CPU (T.zip6 x1 x2 x3 x4 x5 x6)) H.=== [(1, 2, 3, 4, 5, 6), (-1, -2, -3, -4, -5, -6)]
  T.toList (T.run T.CPU (T.zipWith6 (\a1 a2 a3 a4 a5 a6 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6) x1 x2 x3 x4 x5 x6)) H.=== [91, -91]

propZip7 :: H.Property
propZip7 = H.property $ do
  let x1 = T.fromList @'[2] [1, -1 :: Int]
      x2 = T.fromList @'[2] [2, -2 :: Int]
      x3 = T.fromList @'[2] [3, -3 :: Int]
      x4 = T.fromList @'[2] [4, -4 :: Int]
      x5 = T.fromList @'[2] [5, -5 :: Int]
      x6 = T.fromList @'[2] [6, -6 :: Int]
      x7 = T.fromList @'[2] [7, -7 :: Int]
  T.toList (T.run T.CPU (T.zip7 x1 x2 x3 x4 x5 x6 x7)) H.=== [(1, 2, 3, 4, 5, 6, 7), (-1, -2, -3, -4, -5, -6, -7)]
  T.toList (T.run T.CPU (T.zipWith7 (\a1 a2 a3 a4 a5 a6 a7 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 + 7 * a7) x1 x2 x3 x4 x5 x6 x7)) H.=== [140, -140]

propZip8 :: H.Property
propZip8 = H.property $ do
  let x1 = T.fromList @'[2] [1, -1 :: Int]
      x2 = T.fromList @'[2] [2, -2 :: Int]
      x3 = T.fromList @'[2] [3, -3 :: Int]
      x4 = T.fromList @'[2] [4, -4 :: Int]
      x5 = T.fromList @'[2] [5, -5 :: Int]
      x6 = T.fromList @'[2] [6, -6 :: Int]
      x7 = T.fromList @'[2] [7, -7 :: Int]
      x8 = T.fromList @'[2] [8, -8 :: Int]
  T.toList (T.run T.CPU (T.zip8 x1 x2 x3 x4 x5 x6 x7 x8)) H.=== [(1, 2, 3, 4, 5, 6, 7, 8), (-1, -2, -3, -4, -5, -6, -7, -8)]
  T.toList (T.run T.CPU (T.zipWith8 (\a1 a2 a3 a4 a5 a6 a7 a8 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 + 7 * a7 + 8 * a8) x1 x2 x3 x4 x5 x6 x7 x8)) H.=== [204, -204]

propZip9 :: H.Property
propZip9 = H.property $ do
  let x1 = T.fromList @'[2] [1, -1 :: Int]
      x2 = T.fromList @'[2] [2, -2 :: Int]
      x3 = T.fromList @'[2] [3, -3 :: Int]
      x4 = T.fromList @'[2] [4, -4 :: Int]
      x5 = T.fromList @'[2] [5, -5 :: Int]
      x6 = T.fromList @'[2] [6, -6 :: Int]
      x7 = T.fromList @'[2] [7, -7 :: Int]
      x8 = T.fromList @'[2] [8, -8 :: Int]
      x9 = T.fromList @'[2] [9, -9 :: Int]
  T.toList (T.run T.CPU (T.zip9 x1 x2 x3 x4 x5 x6 x7 x8 x9)) H.=== [(1, 2, 3, 4, 5, 6, 7, 8, 9), (-1, -2, -3, -4, -5, -6, -7, -8, -9)]
  T.toList (T.run T.CPU (T.zipWith9 (\a1 a2 a3 a4 a5 a6 a7 a8 a9 -> a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 + 7 * a7 + 8 * a8 + 9 * a9) x1 x2 x3 x4 x5 x6 x7 x8 x9)) H.=== [285, -285]

propNineEdges :: H.Property
propNineEdges = H.property $ do
  let integers = T.fromList @'[2] [1, 2 :: Int]
      flags = T.fromList @'[2] [True, False]
      floats = T.full @'[2] (0.5 :: Double)
      empty = T.zeros @'[0] @Int
      scalar = T.scalar (T.constant (1 :: Int))
      add9 a b c d e f g h i = a + b + c + d + e + f + g + h + i
      mixed = T.zip9 integers flags floats integers integers integers integers integers integers
      selected = T.zipWith9 (\a flag c d e f g h i -> T.select flag (T.fromIntegral (a + d + e + f + g + h + i) + c) 0) integers flags floats integers integers integers integers integers integers
  T.toList (T.run T.CPU mixed) H.=== [(1, True, 0.5, 1, 1, 1, 1, 1, 1), (2, False, 0.5, 2, 2, 2, 2, 2, 2)]
  T.toList (T.run T.CPU selected) H.=== [7.5, 0]
  T.toList (T.run T.CPU (T.zip9 empty empty empty empty empty empty empty empty empty)) H.=== []
  T.toList (T.run T.CPU (T.zipWith9 add9 empty empty empty empty empty empty empty empty empty)) H.=== []
  T.toList (T.run T.CPU (T.zip9 scalar scalar scalar scalar scalar scalar scalar scalar scalar)) H.=== [(1, 1, 1, 1, 1, 1, 1, 1, 1)]
  T.toList (T.run T.CPU (T.zipWith9 add9 scalar scalar scalar scalar scalar scalar scalar scalar scalar)) H.=== [9]

propUnzip2 :: H.Property
propUnzip2 = H.property $ do
  let input = T.fromList @'[2] [(1, 2), (-1, -2)] :: T.Tensor '[2] (Int, Int)
      (x1, x2) = T.unzip input
  T.toList (T.run T.CPU x1) H.=== [1, -1]
  T.toList (T.run T.CPU x2) H.=== [2, -2]
  T.toList (T.run T.CPU (T.zip x1 x2)) H.=== T.toList (T.run T.CPU input)

propUnzip3 :: H.Property
propUnzip3 = H.property $ do
  let input = T.fromList @'[2] [(1, 2, 3), (-1, -2, -3)] :: T.Tensor '[2] (Int, Int, Int)
      (x1, x2, x3) = T.unzip3 input
  T.toList (T.run T.CPU x1) H.=== [1, -1]
  T.toList (T.run T.CPU x2) H.=== [2, -2]
  T.toList (T.run T.CPU x3) H.=== [3, -3]
  T.toList (T.run T.CPU (T.zip3 x1 x2 x3)) H.=== T.toList (T.run T.CPU input)

propUnzip4 :: H.Property
propUnzip4 = H.property $ do
  let input = T.fromList @'[2] [(1, 2, 3, 4), (-1, -2, -3, -4)] :: T.Tensor '[2] (Int, Int, Int, Int)
      (x1, x2, x3, x4) = T.unzip4 input
  T.toList (T.run T.CPU x1) H.=== [1, -1]
  T.toList (T.run T.CPU x2) H.=== [2, -2]
  T.toList (T.run T.CPU x3) H.=== [3, -3]
  T.toList (T.run T.CPU x4) H.=== [4, -4]
  T.toList (T.run T.CPU (T.zip4 x1 x2 x3 x4)) H.=== T.toList (T.run T.CPU input)

propUnzip5 :: H.Property
propUnzip5 = H.property $ do
  let input = T.fromList @'[2] [(1, 2, 3, 4, 5), (-1, -2, -3, -4, -5)] :: T.Tensor '[2] (Int, Int, Int, Int, Int)
      (x1, x2, x3, x4, x5) = T.unzip5 input
  T.toList (T.run T.CPU x1) H.=== [1, -1]
  T.toList (T.run T.CPU x2) H.=== [2, -2]
  T.toList (T.run T.CPU x3) H.=== [3, -3]
  T.toList (T.run T.CPU x4) H.=== [4, -4]
  T.toList (T.run T.CPU x5) H.=== [5, -5]
  T.toList (T.run T.CPU (T.zip5 x1 x2 x3 x4 x5)) H.=== T.toList (T.run T.CPU input)

propUnzip6 :: H.Property
propUnzip6 = H.property $ do
  let input = T.fromList @'[2] [(1, 2, 3, 4, 5, 6), (-1, -2, -3, -4, -5, -6)] :: T.Tensor '[2] (Int, Int, Int, Int, Int, Int)
      (x1, x2, x3, x4, x5, x6) = T.unzip6 input
  T.toList (T.run T.CPU x1) H.=== [1, -1]
  T.toList (T.run T.CPU x2) H.=== [2, -2]
  T.toList (T.run T.CPU x3) H.=== [3, -3]
  T.toList (T.run T.CPU x4) H.=== [4, -4]
  T.toList (T.run T.CPU x5) H.=== [5, -5]
  T.toList (T.run T.CPU x6) H.=== [6, -6]
  T.toList (T.run T.CPU (T.zip6 x1 x2 x3 x4 x5 x6)) H.=== T.toList (T.run T.CPU input)

propUnzip7 :: H.Property
propUnzip7 = H.property $ do
  let input = T.fromList @'[2] [(1, 2, 3, 4, 5, 6, 7), (-1, -2, -3, -4, -5, -6, -7)] :: T.Tensor '[2] (Int, Int, Int, Int, Int, Int, Int)
      (x1, x2, x3, x4, x5, x6, x7) = T.unzip7 input
  T.toList (T.run T.CPU x1) H.=== [1, -1]
  T.toList (T.run T.CPU x2) H.=== [2, -2]
  T.toList (T.run T.CPU x3) H.=== [3, -3]
  T.toList (T.run T.CPU x4) H.=== [4, -4]
  T.toList (T.run T.CPU x5) H.=== [5, -5]
  T.toList (T.run T.CPU x6) H.=== [6, -6]
  T.toList (T.run T.CPU x7) H.=== [7, -7]
  T.toList (T.run T.CPU (T.zip7 x1 x2 x3 x4 x5 x6 x7)) H.=== T.toList (T.run T.CPU input)

propUnzip8 :: H.Property
propUnzip8 = H.property $ do
  let input = T.fromList @'[2] [(1, 2, 3, 4, 5, 6, 7, 8), (-1, -2, -3, -4, -5, -6, -7, -8)] :: T.Tensor '[2] (Int, Int, Int, Int, Int, Int, Int, Int)
      (x1, x2, x3, x4, x5, x6, x7, x8) = T.unzip8 input
  T.toList (T.run T.CPU x1) H.=== [1, -1]
  T.toList (T.run T.CPU x2) H.=== [2, -2]
  T.toList (T.run T.CPU x3) H.=== [3, -3]
  T.toList (T.run T.CPU x4) H.=== [4, -4]
  T.toList (T.run T.CPU x5) H.=== [5, -5]
  T.toList (T.run T.CPU x6) H.=== [6, -6]
  T.toList (T.run T.CPU x7) H.=== [7, -7]
  T.toList (T.run T.CPU x8) H.=== [8, -8]
  T.toList (T.run T.CPU (T.zip8 x1 x2 x3 x4 x5 x6 x7 x8)) H.=== T.toList (T.run T.CPU input)

propUnzip9 :: H.Property
propUnzip9 = H.property $ do
  let input = T.fromList @'[2] [(1, 2, 3, 4, 5, 6, 7, 8, 9), (-1, -2, -3, -4, -5, -6, -7, -8, -9)] :: T.Tensor '[2] (Int, Int, Int, Int, Int, Int, Int, Int, Int)
      (x1, x2, x3, x4, x5, x6, x7, x8, x9) = T.unzip9 input
  T.toList (T.run T.CPU x1) H.=== [1, -1]
  T.toList (T.run T.CPU x2) H.=== [2, -2]
  T.toList (T.run T.CPU x3) H.=== [3, -3]
  T.toList (T.run T.CPU x4) H.=== [4, -4]
  T.toList (T.run T.CPU x5) H.=== [5, -5]
  T.toList (T.run T.CPU x6) H.=== [6, -6]
  T.toList (T.run T.CPU x7) H.=== [7, -7]
  T.toList (T.run T.CPU x8) H.=== [8, -8]
  T.toList (T.run T.CPU x9) H.=== [9, -9]
  T.toList (T.run T.CPU (T.zip9 x1 x2 x3 x4 x5 x6 x7 x8 x9)) H.=== T.toList (T.run T.CPU input)

propUnzipTypes :: H.Property
propUnzipTypes = H.property $ do
  let mixed = T.fromList @'[2] [(1 :: Int, True, 0.5 :: Double), (2, False, 1.5)]
      (integers, flags, floats) = T.unzip3 mixed
      nested = T.fromList @'[2] [((3 :: Int, True), 2.5 :: Double), ((4, False), 3.5)]
      (pairs, values) = T.unzip nested
      (nestedIntegers, nestedFlags) = T.unzip pairs
  T.toList (T.run T.CPU integers) H.=== [1, 2]
  T.toList (T.run T.CPU flags) H.=== [True, False]
  T.toList (T.run T.CPU floats) H.=== [0.5, 1.5]
  T.toList (T.run T.CPU nestedIntegers) H.=== [3, 4]
  T.toList (T.run T.CPU nestedFlags) H.=== [True, False]
  T.toList (T.run T.CPU values) H.=== [2.5, 3.5]
  T.toList (T.run T.CPU (T.select flags integers 0)) H.=== [1, 0]

propUnzipEdges :: H.Property
propUnzipEdges = H.property $ do
  let (scalarInt, scalarBool) = T.unzip (T.scalar (T.constant (7 :: Int, True)))
      scalar = T.scalar (T.constant (2 :: Int))
      (a, b, c, d, e, f, g, h, i) = T.unzip9 (T.zip9 scalar scalar scalar scalar scalar scalar scalar scalar scalar)
      empty = T.zeros @'[2, 0, 3] @Int
      (a0, b0, c0, d0, e0, f0, g0, h0, i0) = T.unzip9 (T.zip9 empty empty empty empty empty empty empty empty empty)
  T.toList (T.run T.CPU scalarInt) H.=== [7]
  T.toList (T.run T.CPU scalarBool) H.=== [True]
  map (T.toList . T.run T.CPU) [a, b, c, d, e, f, g, h, i] H.=== replicate 9 [2]
  map (T.toList . T.run T.CPU) [a0, b0, c0, d0, e0, f0, g0, h0, i0] H.=== replicate 9 []
