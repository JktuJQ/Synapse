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
values = T.toList . T.run T.CPU

chunks :: Int -> [a] -> [[a]]
chunks _ [] = []
chunks size xs = take size xs : chunks size (drop size xs)

reference :: (Num a) => Int -> Int -> [a] -> [a] -> [a]
reference shared cols lhs rhs =
  [sum (zipWith (*) row col) | row <- chunks shared lhs, col <- transpose (chunks cols rhs)]

propDot :: H.Property
propDot = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 4) (Gen.int (Range.linear (-20) 20))
  ys <- H.forAll $ Gen.list (Range.singleton 4) (Gen.int (Range.linear (-20) 20))
  values (T.dot (T.fromList @'[4] xs) (T.fromList @'[4] ys)) H.=== [sum (zipWith (*) xs ys)]

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
  approxList (values (T.dot (T.fromList @'[3] [1, 2, 3 :: Float]) (T.fromList @'[3] [4, 5, 6]))) [32]
  values (T.sum (T.matmul lhs rhs)) H.=== [163]
  values (T.fromGen @'[2] (\ix -> T.index @'[] (T.dot (T.ones @'[3] @Int) (T.ones @'[3])) + T.coordinate @0 ix)) H.=== [3, 4]
  values (T.matmul (T.ones @'[1, 1] @Float) (T.ones @'[1, 1])) H.=== [1]
  values (T.matmul (T.ones @'[2, 3] @Float) (T.ones @'[3])) H.=== [3, 3]

propEmpty :: H.Property
propEmpty = H.property $ do
  values (T.dot (T.zeros @'[0] @Int) (T.zeros @'[0])) H.=== [0]
  values (T.matmul (T.zeros @'[2, 0] @Int) (T.zeros @'[0, 3])) H.=== replicate 6 0
  values (T.matmul (T.zeros @'[2, 3, 0] @Double) (T.zeros @'[2, 0, 4])) H.=== replicate 24 0
  values (T.matmul (T.zeros @'[0, 2, 3] @Int) (T.zeros @'[0, 3, 4])) H.=== []
  values (T.matmul (T.zeros @'[0, 3] @Int) (T.zeros @'[3, 4])) H.=== []
  values (T.matmul (T.zeros @'[2, 3] @Int) (T.zeros @'[3, 0])) H.=== []
  values (T.matmul (T.zeros @'[0] @Int) (T.zeros @'[0])) H.=== [0]
  values (T.matmul (T.zeros @'[0] @Int) (T.zeros @'[0, 3])) H.=== [0, 0, 0]
  values (T.matmul (T.zeros @'[2, 0] @Int) (T.zeros @'[0])) H.=== [0, 0]
  values (T.matmul (T.ones @'[1, 2, 3] @Int) (T.zeros @'[0, 3, 4])) H.=== []
  values (T.matmul (T.zeros @'[0, 2, 3] @Int) (T.ones @'[1, 3, 4])) H.=== []

propShapes :: H.Property
propShapes = H.property $ do
  T.shape @(M.MatmulShape '[2, 3] '[3, 4]) H.=== [2, 4]
  T.shape @(M.MatmulShape '[8, 2, 3] '[8, 3, 4]) H.=== [8, 2, 4]
  T.shape @(M.MatmulShape '[2, 8, 2, 3] '[2, 8, 3, 4]) H.=== [2, 8, 2, 4]
  T.shape @(M.MatmulShape '[2, 0] '[0, 4]) H.=== [2, 4]
  T.shape @(M.MatmulShape '[3] '[3]) H.=== []
  T.shape @(M.MatmulShape '[3] '[3, 4]) H.=== [4]
  T.shape @(M.MatmulShape '[2, 3] '[3]) H.=== [2]
  T.shape @(M.MatmulShape '[2, 3] '[8, 3, 4]) H.=== [8, 2, 4]
  T.shape @(M.MatmulShape '[3] '[8, 3, 4]) H.=== [8, 4]
  T.shape @(M.MatmulShape '[8, 2, 3] '[3]) H.=== [8, 2]
  T.shape @(M.MatmulShape '[2, 1, 3, 4] '[5, 4, 6]) H.=== [2, 5, 3, 6]
  T.shape @(M.MatmulShape '[0, 2, 3] '[1, 3, 4]) H.=== [0, 2, 4]
  T.shape @(M.MatmulShape '[1, 2, 3] '[0, 3, 4]) H.=== [0, 2, 4]

propVectors :: H.Property
propVectors = H.withTests 20 $ H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 3) (Gen.int (Range.linear (-10) 10))
  ys <- H.forAll $ Gen.list (Range.singleton 3) (Gen.int (Range.linear (-10) 10))
  matrix <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-10) 10))
  values (T.matmul (T.fromList @'[3] xs) (T.fromList @'[3] ys)) H.=== [sum (zipWith (*) xs ys)]
  values (T.matmul (T.fromList @'[3] xs) (T.fromList @'[3, 2] matrix)) H.=== reference 3 2 xs matrix
  values (T.matmul (T.fromList @'[2, 3] matrix) (T.fromList @'[3] ys)) H.=== reference 3 1 matrix ys

propSharedBatches :: H.Property
propSharedBatches = H.withTests 20 $ H.property $ do
  batches <- H.forAll $ Gen.list (Range.singleton 12) (Gen.int (Range.linear (-10) 10))
  matrix <- H.forAll $ Gen.list (Range.singleton 6) (Gen.int (Range.linear (-10) 10))
  values (T.matmul (T.fromList @'[2, 2, 3] batches) (T.fromList @'[3, 2] matrix)) H.=== concatMap (\batch -> reference 3 2 batch matrix) (chunks 6 batches)
  values (T.matmul (T.fromList @'[2, 3] matrix) (T.fromList @'[2, 3, 2] batches)) H.=== concatMap (reference 3 2 matrix) (chunks 6 batches)

propBatchedVectors :: H.Property
propBatchedVectors = H.withTests 20 $ H.property $ do
  vector <- H.forAll $ Gen.list (Range.singleton 3) (Gen.int (Range.linear (-10) 10))
  batches <- H.forAll $ Gen.list (Range.singleton 12) (Gen.int (Range.linear (-10) 10))
  values (T.matmul (T.fromList @'[3] vector) (T.fromList @'[2, 3, 2] batches)) H.=== concatMap (reference 3 2 vector) (chunks 6 batches)
  values (T.matmul (T.fromList @'[2, 2, 3] batches) (T.fromList @'[3] vector)) H.=== concatMap (\batch -> reference 3 1 batch vector) (chunks 6 batches)

propBroadcastBatches :: H.Property
propBroadcastBatches = H.withTests 20 $ H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 12) (Gen.int (Range.linear (-10) 10))
  ys <- H.forAll $ Gen.list (Range.singleton 18) (Gen.int (Range.linear (-10) 10))
  let expected = concat [reference 3 2 left right | left <- chunks 6 xs, right <- chunks 6 ys]
  values (T.matmul (T.fromList @'[2, 1, 2, 3] xs) (T.fromList @'[3, 3, 2] ys)) H.=== expected
  values (T.matmul (T.fromList @'[2, 1, 2, 3] xs) (T.fromList @'[1, 3, 3, 2] ys)) H.=== expected

tests :: TestTree
tests =
  testGroup "linear algebra" $
    [ testProperty "dot agrees with vector inner products" propDot,
      testProperty "matrix products agree with a list reference" propMatmul,
      testProperty "rectangular products pack complete columns" propRectangular,
      testProperty "matrix-vector products preserve the contiguous input" propColumn,
      testProperty "multiple batch dimensions remain independent" propBatched,
      testProperty "vectors support scalar, row, and column products" propVectors,
      testProperty "matrix batches share either operand" propSharedBatches,
      testProperty "vectors broadcast across matrix batches" propBatchedVectors,
      testProperty "singleton batch dimensions broadcast on both sides" propBroadcastBatches,
      testProperty "floating products compose with tensor computations" (H.withTests 1 propFloating),
      testProperty "zero dimensions and empty vectors are supported" (H.withTests 1 propEmpty),
      testProperty "output shapes preserve batches" (H.withTests 1 propShapes),
      testProperty "valid linear algebra types compile" (H.withTests 1 (Compiler.accepts (root ++ "Accepted/Linear.hs"))),
      testProperty "vector products compile" (H.withTests 1 (Compiler.accepts (root ++ "Accepted/MatmulVector.hs")))
    ]
      ++ [ testProperty file (H.withTests 1 (Compiler.rejects (root ++ "Rejected/" ++ file ++ ".hs") diagnostic))
           | (file, diagnostic) <-
               [ ("MatmulInner", "Matmul inner dimensions must match"),
                 ("MatmulBatch", "Cannot broadcast dimensions"),
                 ("MatmulBatchRank", "Cannot broadcast dimensions"),
                 ("MatmulVectorInner", "Matmul inner dimensions must match"),
                 ("MatmulScalar", "Matmul requires tensors of rank at least one"),
                 ("MatmulRightScalar", "Matmul requires tensors of rank at least one"),
                 ("MatmulEmptyBatch", "Cannot broadcast dimensions"),
                 ("DotLength", "Couldn't match type"),
                 ("DotMatrix", "Couldn't match type")
               ]
         ]
  where
    root = "test/TensorTest/Ops/Math/Fixtures/"
