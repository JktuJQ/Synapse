{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module TensorTest.Ops.Convolution (tests) where

import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import Synapse.Tensor.Shapes.Convolution (Conv2dShape)
import qualified TensorTest.Compiler as Compiler
import TensorTest.Ops.Convolution.Reference (reference)
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

check :: forall stride padding input kernel. (T.Shape stride, T.Shape padding, T.Shape input, T.Shape kernel, T.Shape (Conv2dShape input kernel stride padding '[1, 1] 1)) => H.Property
check = H.withTests 10 $ H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton $ T.shapeSize @input) $ Gen.int $ Range.linear (-5) 5
  ws <- H.forAll $ Gen.list (Range.singleton $ T.shapeSize @kernel) $ Gen.int $ Range.linear (-5) 5
  let output = T.conv2d @stride @padding @'[1, 1] @1 (T.fromList @input xs) (T.fromList @kernel ws)
  T.toList (T.run T.CPU output) H.=== reference (T.shape @input) (T.shape @kernel) (T.shape @stride) (T.shape @padding) xs ws

propDTypes :: H.Property
propDTypes = H.withTests 1 $ H.property $ do
  let input = T.fromList @'[1, 1, 2, 2] [1, 2, 3, 4 :: Float]
      weights = T.fromList @'[1, 1, 1, 1] [0.5]
  approxList (T.toList $ T.run T.CPU $ T.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] @1 input weights) [0.5, 1, 1.5, 2]

propEmpty :: H.Property
propEmpty = H.withTests 1 $ H.property $ do
  T.toList (T.run T.CPU $ T.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] @1 (T.zeros @'[0, 2, 3, 3] @Int) (T.ones @'[4, 2, 2, 2])) H.=== []
  T.toList (T.run T.CPU $ T.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] @1 (T.ones @'[2, 2, 3, 3] @Int) (T.zeros @'[0, 2, 2, 2])) H.=== []
  T.toList (T.run T.CPU $ T.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] @1 (T.zeros @'[1, 0, 3, 3] @Int) (T.zeros @'[2, 0, 2, 2])) H.=== replicate 8 0
  T.toList (T.run T.CPU $ T.conv2d @'[1, 1] @'[1, 1, 1, 1] @'[1, 1] @1 (T.zeros @'[1, 1, 0, 0] @Int) (T.ones @'[2, 1, 1, 1])) H.=== replicate 8 0

propPrepared :: H.Property
propPrepared = H.withTests 1 $ H.property $ do
  let execute = T.run2 T.CPU $ T.conv2d @'[2, 1] @'[1, 0, 0, 1] @'[1, 1] @1
      input = T.fromListData @'[2, 2, 3, 4] [1 .. 48 :: Double]
      weights = T.fromListData @'[3, 2, 2, 3] [1 .. 36 :: Double]
      expected = reference [2, 2, 3, 4] [3, 2, 2, 3] [2, 1] [1, 0, 0, 1] [1 .. 48] [1 .. 36]
  approxList (T.toList $ execute input weights) expected
  approxList (T.toList $ execute (T.fromListData [2 .. 49]) (T.fromListData [2 .. 37])) $ reference [2, 2, 3, 4] [3, 2, 2, 3] [2, 1] [1, 0, 0, 1] [2 .. 49] [2 .. 37]

tests :: TestTree
tests =
  testGroup "convolution" $
    [ testProperty "unflipped rectangular kernels and multiple batches" $ check @'[1, 1] @'[0, 0, 0, 0] @'[2, 2, 3, 4] @'[3, 2, 2, 3],
      testProperty "same padding and independent channels" $ check @'[1, 1] @'[1, 1, 1, 1] @'[1, 2, 4, 5] @'[3, 2, 3, 3],
      testProperty "unequal strides and asymmetric padding" $ check @'[2, 3] @'[2, 0, 1, 2] @'[2, 2, 4, 5] @'[3, 2, 3, 2],
      testProperty "padding larger than the input" $ check @'[3, 2] @'[3, 2, 4, 1] @'[1, 1, 1, 2] @'[2, 1, 2, 3],
      testProperty "pointwise convolution" $ check @'[1, 1] @'[0, 0, 0, 0] @'[2, 3, 2, 3] @'[2, 3, 1, 1],
      testProperty "Float elements" propDTypes,
      testProperty "empty batches, channels and padded spatial dimensions" propEmpty,
      testProperty "prepared convolution accepts changing inputs and weights" propPrepared,
      testProperty "valid convolution signatures compile" $ H.withTests 1 $ Compiler.accepts $ root ++ "Accepted/Valid.hs"
    ]
      ++ [testProperty file $ H.withTests 1 $ Compiler.rejects (root ++ "Rejected/" ++ file ++ ".hs") diagnostic | (file, diagnostic) <- [("Channels", "Convolution input and weight channels must match"), ("Kernel", "Window kernel exceeds the padded input"), ("ZeroKernel", "Window kernel dimensions must be positive"), ("ZeroStride", "Window strides must be positive"), ("Rank", "Conv2d requires NCHW inputs"), ("StrideRank", "Conv2d requires NCHW inputs"), ("PaddingRank", "Conv2d requires NCHW inputs"), ("Output", "Couldn't match type")]]
  where
    root = "test/TensorTest/Ops/Convolution/Fixtures/"
