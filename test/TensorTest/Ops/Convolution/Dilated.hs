{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Convolution.Dilated (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import qualified TensorTest.Ops.Spatial.Reference as R
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

propGrouped :: H.Property
propGrouped = H.withTests 1 $ H.property $ do
  let one = T.conv1d @'[2] @'[1, 1] @'[2] @2 (T.fromList @'[2, 4, 5] [1 .. 40 :: Double]) (T.fromList @'[6, 2, 2] [1 .. 24])
      two = T.conv2d @'[1, 2] @'[1, 0, 1, 1] @'[2, 1] @2 (T.fromList @'[1, 4, 4, 5] [1 .. 80 :: Double]) (T.fromList @'[6, 2, 2, 3] [1 .. 72])
      three = T.conv3d @'[1, 1, 2] @'[1, 0, 1, 1, 0, 1] @'[1, 2, 1] @2 (T.fromList @'[1, 4, 2, 3, 3] [1 .. 72 :: Double]) (T.fromList @'[4, 2, 2, 2, 2] [1 .. 64])
      depthwise = T.conv1d @'[1] @'[1, 1] @'[2] @3 (T.fromList @'[1, 3, 3] [1 .. 9 :: Float]) (T.fromList @'[6, 1, 2] [1 .. 12])
  approxList (T.toList $ T.run T.CPU one) $ R.convWith [2, 4, 5] [6, 2, 2] [2] [1, 1] [2] 2 [1 .. 40] [1 .. 24]
  approxList (T.toList $ T.run T.CPU two) $ R.convWith [1, 4, 4, 5] [6, 2, 2, 3] [1, 2] [1, 0, 1, 1] [2, 1] 2 [1 .. 80] [1 .. 72]
  approxList (T.toList $ T.run T.CPU three) $ R.convWith [1, 4, 2, 3, 3] [4, 2, 2, 2, 2] [1, 1, 2] [1, 0, 1, 1, 0, 1] [1, 2, 1] 2 [1 .. 72] [1 .. 64]
  approxList (T.toList $ T.run T.CPU depthwise) $ R.convWith [1, 3, 3] [6, 1, 2] [1] [1, 1] [2] 3 [1 .. 9] [1 .. 12]
  T.toList (T.run T.CPU $ T.conv1d @'[1] @'[0, 0] @'[1] @2 (T.zeros @'[1, 0, 2] @Int) (T.zeros @'[2, 0, 1])) H.=== replicate 4 0

propTranspose :: H.Property
propTranspose = H.withTests 1 $ H.property $ do
  let one = T.convTranspose1d @'[2] @'[1, 0] @'[2] @2 @'[1] (T.fromList @'[1, 4, 2] [1 .. 8 :: Double]) (T.fromList @'[4, 3, 2] [1 .. 24])
      two = T.convTranspose2d @'[2, 1] @'[1, 0, 0, 1] @'[2, 1] @2 @'[1, 0] (T.fromList @'[1, 4, 2, 2] [1 .. 16 :: Double]) (T.fromList @'[4, 3, 2, 2] [1 .. 48])
      three = T.convTranspose3d @'[1, 2, 2] @'[0, 1, 0, 0, 1, 0] @'[2, 1, 2] @2 @'[0, 1, 1] (T.fromList @'[1, 4, 1, 2, 2] [1 .. 16 :: Double]) (T.fromList @'[4, 1, 2, 1, 2] [1 .. 16])
  approxList (T.toList $ T.run T.CPU one) $ R.transposedWith [1, 4, 2] [4, 3, 2] [2] [1, 0] [2] 2 [1] [1 .. 8] [1 .. 24]
  approxList (T.toList $ T.run T.CPU two) $ R.transposedWith [1, 4, 2, 2] [4, 3, 2, 2] [2, 1] [1, 0, 0, 1] [2, 1] 2 [1, 0] [1 .. 16] [1 .. 48]
  approxList (T.toList $ T.run T.CPU three) $ R.transposedWith [1, 4, 1, 2, 2] [4, 1, 2, 1, 2] [1, 2, 2] [0, 1, 0, 0, 1, 0] [2, 1, 2] 2 [0, 1, 1] [1 .. 16] [1 .. 16]

propPatches :: H.Property
propPatches = H.withTests 1 $ H.property $ do
  let input = T.fromList @'[1, 2, 3, 4] [1 .. 24 :: Double]
      extracted = T.extractPatches2d @'[2, 2] @'[1, 2] @'[1, 0, 1, 1] @'[2, 1] input
      expected = R.patchesWith [1, 2, 3, 4] [2, 2] [1, 2] [1, 0, 1, 1] [2, 1] [1 .. 24]
      combined = T.combinePatches2d @'[1, 2, 3, 4] @'[2, 2] @'[1, 2] @'[1, 0, 1, 1] @'[2, 1] extracted
  approxList (T.toList $ T.run T.CPU extracted) expected
  approxList (T.toList $ T.run T.CPU combined) $ R.combineWith [1, 2, 3, 4] [2, 2] [1, 2] [1, 0, 1, 1] [2, 1] expected

propPooling :: H.Property
propPooling = H.withTests 1 $ H.property $ do
  let input = T.fromList @'[1, 1, 4, 4] [-16 .. -1 :: Double]
      result = T.maxPool2d @'[2, 2] @'[1, 1] @'[1, 1, 1, 1] @'[2, 2] input
      (empty, gradient) = AG.backprop (T.maxPool1d @'[2] @'[1] @'[1, 1] @'[2]) $ T.fromList @'[1, 1, 1] [5 :: Double]
  approxList (T.toList $ T.run T.CPU result) $ R.poolWith [1, 1, 4, 4] [2, 2] [1, 1] [1, 1, 1, 1] [2, 2] maximum [-16 .. -1]
  T.toList (T.run T.CPU empty) H.=== [-(1 / 0)]
  T.toList (T.run T.CPU gradient) H.=== [0]
  T.toList (T.run T.CPU $ T.maxPool1d @'[2] @'[1] @'[1, 1] @'[2] $ T.fromList @'[1, 1, 1] [5 :: Int]) H.=== [minBound]

propPrepared :: H.Property
propPrepared = H.withTests 1 $ H.property $ do
  let execute = T.run2 T.CPU $ T.conv1d @'[1] @'[0, 0] @'[2] @2
      first = execute (T.fromListData @'[1, 2, 3] [1 .. 6 :: Double]) (T.fromListData @'[2, 1, 2] [1 .. 4])
      second = execute (T.fromListData [2 .. 7]) (T.fromListData [2 .. 5])
  T.toList first H.=== [7, 36]
  T.toList second H.=== [16, 55]

tests :: TestTree
tests =
  testGroup "dilation and groups" $
    [ testProperty "grouped and depthwise convolutions agree with references" propGrouped,
      testProperty "grouped transposed convolutions agree with references" propTranspose,
      testProperty "dilated patch extraction and overlap summation" propPatches,
      testProperty "dilated max pooling handles windows containing only padding" propPooling,
      testProperty "prepared grouped convolution accepts changing values" propPrepared
    ]
      ++ [testProperty file $ H.withTests 1 $ Compiler.rejects (root ++ file ++ ".hs") diagnostic | (file, diagnostic) <- [("ZeroDilation", "Window dilation must be positive"), ("DilationRank", "Conv2d requires"), ("DilatedKernel", "Window kernel exceeds the padded input"), ("GroupsZero", "Convolution groups must be positive"), ("GroupsInput", "must be divisible by groups"), ("GroupsOutput", "must be divisible by groups"), ("GroupsWeight", "within each group"), ("TransposeGroups", "must be divisible by groups"), ("TransposeDilation", "Window dilation must be positive"), ("PatchDilation", "Window dilation must be positive"), ("PoolDilation", "Window dilation must be positive")]]
  where
    root = "test/TensorTest/Ops/Convolution/Fixtures/Rejected/"
