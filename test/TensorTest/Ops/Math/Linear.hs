{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Math.Linear (tests) where

import Data.List (transpose)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Matmul as M
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh, T.DType a) => T.Tensor sh a -> [a]
values = T.toList . T.runCPU

chunks :: Int -> [a] -> [[a]]
chunks _ [] = []
chunks size xs = take size xs : chunks size (drop size xs)

reference :: Num a => Int -> Int -> [a] -> [a] -> [a]
reference shared cols lhs rhs =
  [sum (zipWith (*) row col) | row <- chunks shared lhs, col <- transpose (chunks cols rhs)]

tests :: TestTree
tests = testGroup "linear algebra" $
  [ testProperty "dot agrees with vector inner products" propDot,
    testProperty "matrix products agree with a list reference" propMatmul,
    testProperty "rectangular products pack complete columns" propRectangular,
    testProperty "matrix-vector products preserve the contiguous input" propColumn,
    testProperty "multiple batch dimensions remain independent" propBatched,
    testProperty "floating products compose with tensor computations" (H.withTests 1 propFloating),
    testProperty "zero dimensions and empty vectors are supported" (H.withTests 1 propEmpty),
    testProperty "output shapes preserve batches" (H.withTests 1 propShapes),
    testProperty "valid linear algebra types compile" (H.withTests 1 (Compiler.accepts (root ++ "Accepted/Linear.hs")))
  ] ++
  [ testProperty file (H.withTests 1 (Compiler.rejects (root ++ "Rejected/" ++ file ++ ".hs") diagnostic))
    | (file, diagnostic) <-
      [ ("MatmulInner", "Matmul inner dimensions must match"),
        ("MatmulBatch", "Matmul batch shapes must match"),
        ("MatmulBatchRank", "Matmul batch shapes must match"),
        ("MatmulVector", "Matmul requires tensors of rank at least two"),
        ("MatmulScalar", "Matmul requires tensors of rank at least two"),
        ("DotLength", "Couldn't match type"),
        ("DotMatrix", "Couldn't match type")
      ]
  ]
  where
    root = "test/TensorTest/Ops/Math/Fixtures/"

propDot :: H.Property
propDot = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 4) (Gen.int (Range.linear (-20) 20))
  ys <- H.forAll $ Gen.list (Range.singleton 4) (Gen.int (Range.linear (-20) 20))
  values (T.scalar (T.dot (T.fromList @'[4] xs) (T.fromList @'[4] ys))) H.=== [sum (zipWith (*) xs ys)]

propMatmul :: H.Property
propMatmul = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-20) 20))
  ys <- H.forAll $ Gen.list (Range.singleton 12) (Gen.int (Range.linear (-20) 20))
  let result = T.matmul (T.fromList @'[2, 3] xs) (T.fromList @'[3, 4] ys)
  values result H.=== reference 3 4 xs ys

propRectangular :: H.Property
propRectangular = H.withTests 20 $ H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 51) (Gen.int (Range.linear (-20) 20))
  ys <- H.forAll $ Gen.list (Range.singleton 85) (Gen.int (Range.linear (-20) 20))
  values (T.matmul (T.fromList @'[3, 17] xs) (T.fromList @'[17, 5] ys)) H.=== reference 17 5 xs ys

propColumn :: H.Property
propColumn = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-20) 20))
  ys <- H.forAll $ Gen.list (Range.singleton 3) (Gen.int (Range.linear (-20) 20))
  values (T.matmul (T.fromList @'[2, 3] xs) (T.fromList @'[3, 1] ys)) H.=== reference 3 1 xs ys

propBatched :: H.Property
propBatched = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-20) 20))
  ys <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-20) 20))
  let result = T.matmul (T.fromList @'[2, 2, 2, 3] xs) (T.fromList @'[2, 2, 3, 2] ys)
      expected = concat (zipWith (reference 3 2) (chunks 6 xs) (chunks 6 ys))
  values result H.=== expected
  let flatBatch = T.matmul (T.fromList @'[4, 2, 3] xs) (T.fromList @'[4, 3, 2] ys)
  values flatBatch H.=== expected

propFloating :: H.Property
propFloating = H.property $ do
  let lhs = T.fromList @'[2, 3] [1 .. 6 :: Double]
      rhs = T.fromList @'[3, 2] [1 .. 6 :: Double]
      bias = T.broadcast @'[2, 2] (T.fromList @'[2] [1, 2 :: Double])
  approxList (values (T.matmul lhs rhs + bias)) [23, 30, 50, 66]
  approxList (values (T.matmul (lhs * 2) rhs)) [44, 56, 98, 128]
  approxList (values (T.scalar (T.dot (T.fromList @'[3] [1, 2, 3 :: Float]) (T.fromList @'[3] [4, 5, 6])))) [32]
  values (T.scalar (T.sum (T.matmul lhs rhs))) H.=== [163]
  values (T.fromGen @'[2] (\ix -> T.dot (T.ones @'[3] @Int) (T.ones @'[3]) + T.coordinate @0 ix)) H.=== [3, 4]
  values (T.matmul (T.ones @'[1, 1] @Float) (T.ones @'[1, 1])) H.=== [1]

propEmpty :: H.Property
propEmpty = H.property $ do
  values (T.scalar (T.dot (T.zeros @'[0] @Int) (T.zeros @'[0]))) H.=== [0]
  values (T.matmul (T.zeros @'[2, 0] @Int) (T.zeros @'[0, 3])) H.=== replicate 6 0
  values (T.matmul (T.zeros @'[2, 3, 0] @Double) (T.zeros @'[2, 0, 4])) H.=== replicate 24 0
  values (T.matmul (T.zeros @'[0, 2, 3] @Int) (T.zeros @'[0, 3, 4])) H.=== []
  values (T.matmul (T.zeros @'[0, 3] @Int) (T.zeros @'[3, 4])) H.=== []
  values (T.matmul (T.zeros @'[2, 3] @Int) (T.zeros @'[3, 0])) H.=== []

propShapes :: H.Property
propShapes = H.property $ do
  T.shape @(M.MatmulShape '[2, 3] '[3, 4]) H.=== [2, 4]
  T.shape @(M.MatmulShape '[8, 2, 3] '[8, 3, 4]) H.=== [8, 2, 4]
  T.shape @(M.MatmulShape '[2, 8, 2, 3] '[2, 8, 3, 4]) H.=== [2, 8, 2, 4]
  T.shape @(M.MatmulShape '[2, 0] '[0, 4]) H.=== [2, 4]
