{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Axes.Operations (tests) where

import qualified Data.Array.Accelerate as A
import Data.List (sortOn)
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
    "operations"
    [ testProperty "squeeze removes the selected singleton axis" (H.withTests 1 propSqueeze),
      testProperty "unsqueeze supports every insertion position" (H.withTests 1 propUnsqueezePositions),
      testProperty "unsqueeze and squeeze round-trip" propUnsqueezeSqueeze,
      testProperty "axis permutation changes shape and element order" propPermute,
      testProperty "inverse axis permutations round-trip" propPermuteInverse,
      testProperty "all rank-three permutations agree with list indexing" propAllPermutations,
      testProperty "axis operations preserve non-numeric elements" (H.withTests 1 propNonNumericAxes),
      testProperty "squeeze preserves other singleton axes" (H.withTests 1 propSqueezeOneAxis),
      testProperty "axis operations support scalars and empty tensors" (H.withTests 1 propAxisEdges)
    ]

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

propPermute :: H.Property
propPermute = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3, 4] xs
      result = T.runCPU (T.permuteAxes @'[2, 0, 1] source)
      expected = [xs !! ((i * 3 + j) * 4 + k) | k <- [0 .. 3], i <- [0 .. 1], j <- [0 .. 2]]
  A.arrayShape (Internal.unTensorData result) H.=== (A.Z A.:. (4 :: Int) A.:. (2 :: Int) A.:. (3 :: Int))
  T.toList result H.=== expected

propPermuteInverse :: H.Property
propPermuteInverse = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3, 4] xs
  T.toList (T.runCPU (T.permuteAxes @'[1, 2, 0] (T.permuteAxes @'[2, 0, 1] source))) H.=== xs
  T.toList (T.runCPU (T.permuteAxes @'[0, 1, 2] source)) H.=== xs

propAllPermutations :: H.Property
propAllPermutations = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 24) (Gen.int (Range.linear (-100) 100))
  let source = T.fromList @'[2, 3, 4] xs
      results =
        [ ([0, 1, 2], T.toList (T.runCPU (T.permuteAxes @'[0, 1, 2] source))),
          ([0, 2, 1], T.toList (T.runCPU (T.permuteAxes @'[0, 2, 1] source))),
          ([1, 0, 2], T.toList (T.runCPU (T.permuteAxes @'[1, 0, 2] source))),
          ([1, 2, 0], T.toList (T.runCPU (T.permuteAxes @'[1, 2, 0] source))),
          ([2, 0, 1], T.toList (T.runCPU (T.permuteAxes @'[2, 0, 1] source))),
          ([2, 1, 0], T.toList (T.runCPU (T.permuteAxes @'[2, 1, 0] source)))
        ]
      expected order =
        [ xs !! foldl (\offset (dim, coordinate) -> offset * dim + coordinate) 0 (zip [2, 3, 4] sourceCoordinates)
          | coordinates <- sequence [[0 .. ([2, 3, 4] !! axis) - 1] | axis <- order],
            let sourceCoordinates = map snd (sortOn fst (zip order coordinates))
        ]
  mapM_
    ( \(order, actual) -> do
        H.annotateShow order
        actual H.=== expected order
    )
    results

propNonNumericAxes :: H.Property
propNonNumericAxes = H.property $ do
  let source = T.fromList @'[2, 3] [True, False, True, False, True, False]
  T.toList (T.runCPU (T.permuteAxes @'[1, 0] source)) H.=== [True, False, False, True, True, False]
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
  T.toList (T.runCPU (T.permuteAxes @'[] scalar)) H.=== [7]
  T.toList (T.runCPU (T.permuteAxes @'[2, 0, 1] empty)) H.=== []
  T.toList (T.runCPU (T.squeeze @1 (T.unsqueeze @1 empty))) H.=== []
