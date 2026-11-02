{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Spatial (tests) where

import qualified Hedgehog as H
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import qualified TensorTest.Ops.Convolution.Dilated
import qualified TensorTest.Ops.Spatial.Reference as R
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

propConv :: H.Property
propConv = H.withTests 1 $ H.property $ do
  let xs = [1 .. 20 :: Int]
      ws = [1 .. 12 :: Int]
      one = T.conv1d @'[2] @'[1, 2] @'[1] @1 (T.fromList @'[2, 2, 5] xs) (T.fromList @'[3, 2, 2] ws)
      xs3 = [1 .. 48 :: Int]
      ws3 = [1 .. 24 :: Int]
      three = T.conv3d @'[1, 2, 1] @'[1, 0, 0, 1, 1, 0] @'[1, 1, 1] @1 (T.fromList @'[2, 2, 2, 2, 3] xs3) (T.fromList @'[3, 2, 2, 1, 2] ws3)
  T.toList (T.run T.CPU one) H.=== R.conv [2, 2, 5] [3, 2, 2] [2] [1, 2] xs ws
  T.toList (T.run T.CPU three) H.=== R.conv [2, 2, 2, 2, 3] [3, 2, 2, 1, 2] [1, 2, 1] [1, 0, 0, 1, 1, 0] xs3 ws3
  T.toList (T.run T.CPU $ T.conv1d @'[1] @'[0, 0] @'[1] @1 (T.zeros @'[0, 2, 3] @Int) (T.ones @'[1, 2, 2])) H.=== []
  T.toList (T.run T.CPU $ T.conv3d @'[1, 1, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1] @1 (T.zeros @'[1, 0, 2, 2, 2] @Int) (T.zeros @'[2, 0, 1, 1, 1])) H.=== replicate 16 0

propTranspose :: H.Property
propTranspose = H.withTests 1 $ H.property $ do
  let xs = [1 .. 24 :: Double]
      ws = [1 .. 24 :: Double]
      actual = T.convTranspose2d @'[2, 3] @'[1, 0, 2, 1] @'[1, 1] @1 @'[1, 2] (T.fromList @'[2, 2, 2, 3] xs) (T.fromList @'[2, 3, 2, 2] ws)
  approxList (T.toList $ T.run T.CPU actual) $ R.transposed [2, 2, 2, 3] [2, 3, 2, 2] [2, 3] [1, 0, 2, 1] [1, 2] xs ws
  let ones = T.convTranspose2d @'[2, 2] @'[0, 0, 0, 0] @'[1, 1] @1 @'[1, 1] (T.ones @'[1, 1, 1, 1] @Int) (T.ones @'[1, 1, 2, 2])
  T.toList (T.run T.CPU ones) H.=== [1, 1, 0, 1, 1, 0, 0, 0, 0]

propPatches :: H.Property
propPatches = H.withTests 1 $ H.property $ do
  let xs = [1 .. 24 :: Double]
      extracted = T.extractPatches2d @'[2, 3] @'[1, 2] @'[1, 0, 1, 1] @'[1, 1] $ T.fromList @'[2, 1, 3, 4] xs
      expected = R.patches [2, 1, 3, 4] [2, 3] [1, 2] [1, 0, 1, 1] xs
      combined = T.combinePatches2d @'[2, 1, 3, 4] @'[2, 3] @'[1, 2] @'[1, 0, 1, 1] @'[1, 1] extracted
  approxList (T.toList $ T.run T.CPU extracted) expected
  approxList (T.toList $ T.run T.CPU combined) $ R.combine [2, 1, 3, 4] [2, 3] [1, 2] [1, 0, 1, 1] expected
  let sparse = T.combinePatches2d @'[1, 1, 3, 3] @'[1, 1] @'[2, 2] @'[0, 0, 0, 0] @'[1, 1] $ T.ones @'[1, 1, 4] @Int
  T.toList (T.run T.CPU sparse) H.=== [1, 0, 1, 0, 0, 0, 1, 0, 1]

propPool :: H.Property
propPool = H.withTests 1 $ H.property $ do
  let xs = [-24 .. -1 :: Double]
      input = T.fromList @'[2, 1, 3, 4] xs
      reference = R.pool [2, 1, 3, 4] [2, 3] [1, 2] [1, 0, 1, 1]
  approxList (T.toList $ T.run T.CPU $ T.maxPool2d @'[2, 3] @'[1, 2] @'[1, 0, 1, 1] @'[1, 1] input) $ reference maximum xs
  approxList (T.toList $ T.run T.CPU $ T.avgPool2d @'[2, 3] @'[1, 2] @'[1, 0, 1, 1] True input) $ reference ((/ 6) . sum) xs
  approxList (T.toList $ T.run T.CPU $ T.avgPool2d @'[2, 3] @'[1, 2] @'[1, 0, 1, 1] False input) $ reference (\values -> sum values / fromIntegral (length values)) xs
  T.toList (T.run T.CPU $ T.maxPool2d @'[2, 2] @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] $ T.fromList @'[1, 1, 2, 2] [-4, -3, -2, -1 :: Int]) H.=== [-1]
  let nan = T.maxPool2d @'[2, 2] @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] $ T.fromList @'[1, 1, 2, 2] [0, 0 / 0, 2, 3 :: Float]
  H.assert $ all isNaN $ T.toList $ T.run T.CPU nan
  T.toList (T.run T.CPU $ T.avgPool2d @'[2, 2] @'[1, 1] @'[0, 0, 0, 0] False $ T.zeros @'[0, 2, 2, 2] @Float) H.=== []

propPoolRanks :: H.Property
propPoolRanks = H.withTests 1 $ H.property $ do
  let xs = [-20 .. -1 :: Double]
      input = T.fromList @'[2, 2, 5] xs
      reference = R.pool [2, 2, 5] [3] [2] [1, 1]
      ys = [-48 .. -1 :: Double]
      volume = T.fromList @'[2, 2, 2, 3, 2] ys
      reference3 = R.pool [2, 2, 2, 3, 2] [2, 2, 2] [1, 2, 1] [1, 0, 0, 1, 1, 0]
  approxList (T.toList $ T.run T.CPU $ T.maxPool1d @'[3] @'[2] @'[1, 1] @'[1] input) $ reference maximum xs
  approxList (T.toList $ T.run T.CPU $ T.avgPool1d @'[3] @'[2] @'[1, 1] True input) $ reference ((/ 3) . sum) xs
  approxList (T.toList $ T.run T.CPU $ T.avgPool1d @'[3] @'[2] @'[1, 1] False input) $ reference (\values -> sum values / fromIntegral (length values)) xs
  approxList (T.toList $ T.run T.CPU $ T.maxPool3d @'[2, 2, 2] @'[1, 2, 1] @'[1, 0, 0, 1, 1, 0] @'[1, 1, 1] volume) $ reference3 maximum ys
  approxList (T.toList $ T.run T.CPU $ T.avgPool3d @'[2, 2, 2] @'[1, 2, 1] @'[1, 0, 0, 1, 1, 0] True volume) $ reference3 ((/ 8) . sum) ys
  approxList (T.toList $ T.run T.CPU $ T.avgPool3d @'[2, 2, 2] @'[1, 2, 1] @'[1, 0, 0, 1, 1, 0] False volume) $ reference3 (\values -> sum values / fromIntegral (length values)) ys

propPatchRanks :: H.Property
propPatchRanks = H.withTests 1 $ H.property $ do
  let xs = [1 .. 16 :: Double]
      extracted = T.extractPatches1d @'[2] @'[1] @'[1, 0] @'[1] $ T.fromList @'[2, 2, 4] xs
      expected = R.patches [2, 2, 4] [2] [1] [1, 0] xs
      combined = T.combinePatches1d @'[2, 2, 4] @'[2] @'[1] @'[1, 0] @'[1] extracted
      ys = [1 .. 32 :: Double]
      extracted3 = T.extractPatches3d @'[2, 1, 2] @'[1, 2, 1] @'[1, 0, 0, 0, 1, 0] @'[1, 1, 1] $ T.fromList @'[2, 2, 2, 2, 2] ys
      expected3 = R.patches [2, 2, 2, 2, 2] [2, 1, 2] [1, 2, 1] [1, 0, 0, 0, 1, 0] ys
      combined3 = T.combinePatches3d @'[2, 2, 2, 2, 2] @'[2, 1, 2] @'[1, 2, 1] @'[1, 0, 0, 0, 1, 0] @'[1, 1, 1] extracted3
  approxList (T.toList $ T.run T.CPU extracted) expected
  approxList (T.toList $ T.run T.CPU combined) $ R.combine [2, 2, 4] [2] [1] [1, 0] expected
  approxList (T.toList $ T.run T.CPU extracted3) expected3
  approxList (T.toList $ T.run T.CPU combined3) $ R.combine [2, 2, 2, 2, 2] [2, 1, 2] [1, 2, 1] [1, 0, 0, 0, 1, 0] expected3

propTransposeRanks :: H.Property
propTransposeRanks = H.withTests 1 $ H.property $ do
  let xs = [1 .. 12 :: Double]
      ws = [1 .. 12 :: Double]
      one = T.convTranspose1d @'[3] @'[1, 0] @'[1] @1 @'[2] (T.fromList @'[2, 2, 3] xs) (T.fromList @'[2, 3, 2] ws)
      xs3 = [1 .. 32 :: Double]
      ws3 = [1 .. 24 :: Double]
      three = T.convTranspose3d @'[2, 1, 3] @'[1, 0, 0, 0, 2, 0] @'[1, 1, 1] @1 @'[1, 0, 2] (T.fromList @'[2, 2, 2, 2, 2] xs3) (T.fromList @'[2, 3, 2, 1, 2] ws3)
  approxList (T.toList $ T.run T.CPU one) $ R.transposed [2, 2, 3] [2, 3, 2] [3] [1, 0] [2] xs ws
  approxList (T.toList $ T.run T.CPU three) $ R.transposed [2, 2, 2, 2, 2] [2, 3, 2, 1, 2] [2, 1, 3] [1, 0, 0, 0, 2, 0] [1, 0, 2] xs3 ws3
  T.toList (T.run T.CPU $ T.convTranspose1d @'[1] @'[0, 0] @'[1] @1 @'[0] (T.zeros @'[0, 1, 2] @Int) (T.ones @'[1, 2, 2])) H.=== []
  T.toList (T.run T.CPU $ T.convTranspose1d @'[1] @'[0, 0] @'[1] @1 @'[0] (T.zeros @'[1, 0, 2] @Int) (T.zeros @'[0, 2, 2])) H.=== replicate 6 0
  T.toList (T.run T.CPU $ T.convTranspose3d @'[1, 1, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1] @1 @'[0, 0, 0] (T.ones @'[1, 1, 1, 1, 1] @Int) (T.zeros @'[1, 0, 1, 1, 1])) H.=== []

propTransposePrepared :: H.Property
propTransposePrepared = H.withTests 1 $ H.property $ do
  let execute = T.run2 T.CPU $ T.convTranspose1d @'[2] @'[0, 0] @'[1] @1 @'[1]
      first = execute (T.fromListData @'[1, 1, 2] [1, 2 :: Double]) (T.fromListData @'[1, 1, 2] [3, 4])
      second = execute (T.fromListData [2, 3]) (T.fromListData [1, 2])
  T.toList first H.=== [3, 4, 6, 8, 0]
  T.toList second H.=== [2, 4, 3, 6, 0]

tests :: TestTree
tests =
  testGroup "spatial operations" $
    [ TensorTest.Ops.Convolution.Dilated.tests,
      testProperty "1D and 3D transposed convolution agree with references" propTransposeRanks,
      testProperty "prepared transposed convolution accepts new inputs and weights" propTransposePrepared,
      testProperty "1D and 3D pooling agree with references" propPoolRanks,
      testProperty "1D and 3D patches preserve layout and overlaps" propPatchRanks,
      testProperty "1D and 3D convolution agree with independent references" propConv,
      testProperty "transposed convolution includes output padding" propTranspose,
      testProperty "patch extraction and overlap summation preserve layout" propPatches,
      testProperty "pooling respects negative values, padding and NaN" propPool,
      testProperty "spatial signatures compile" $ H.withTests 1 $ Compiler.accepts $ root ++ "Accepted/Spatial.hs"
    ]
      ++ [ testProperty file $ H.withTests 1 $ Compiler.rejects (root ++ "Rejected/" ++ file ++ ".hs") diagnostic
           | (file, diagnostic) <-
               [ ("Transpose1Padding", "output padding must be smaller than stride"),
                 ("Transpose1Stride", "Window strides must be positive"),
                 ("Transpose3Rank", "ConvTranspose3d requires"),
                 ("Transpose3Channels", "Convolution input and weight channels must match"),
                 ("Transpose3Extra", "ConvTranspose3d requires"),
                 ("Transpose3Size", "output dimensions must be positive"),
                 ("TransposePadding", "output padding must be smaller than stride"),
                 ("TransposeSize", "output dimensions must be positive"),
                 ("PoolPadding", "Pooling padding must not exceed"),
                 ("PoolZero", "Window strides must be positive"),
                 ("PoolEmpty", "Pooling spatial input dimensions must be positive"),
                 ("PatchesShape", "Couldn't match type"),
                 ("Pool1Rank", "Pool1d requires"),
                 ("Pool3Padding", "Pooling padding must not exceed"),
                 ("Patches1Shape", "Couldn't match type"),
                 ("Patches3Stride", "Window strides must be positive"),
                 ("Conv1Rank", "Conv1d requires"),
                 ("Conv3Channels", "Convolution input and weight channels must match")
               ]
         ]
  where
    root = "test/TensorTest/Ops/Convolution/Fixtures/"
