{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module AutogradTest.Tensor.Spatial (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import qualified TensorTest.Ops.Spatial.Reference as R
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

differences :: ([Double] -> Double) -> [Double] -> [Double]
differences f values =
  let perturb i delta = zipWith (\j value -> if i == j then value + delta else value) [0 ..] values
   in [(f (perturb i 1e-5) - f (perturb i (-1e-5))) / 2e-5 | i <- [0 .. length values - 1]]

checkUnary :: forall input out. (T.Shape input, T.Shape out) => (forall s. (AG.Context s) => AG.Tensor s input Double -> AG.Tensor s out Double) -> ([Double] -> [Double]) -> [Double] -> H.PropertyT IO ()
checkUnary f reference xs = do
  let seeds = take (T.shapeSize @out) $ cycle [0.3, -0.5, 0.8, 0.2]
      (output, backward) = AG.pullback f $ T.fromList @input xs
      (actual, gradient) = T.run T.CPU (output, backward $ T.fromList seeds)
      objective = sum . zipWith (*) seeds . reference
  approxList (T.toList actual) $ reference xs
  approxList (T.toList gradient) $ differences objective xs

checkBinary :: forall input kernel out. (T.Shape input, T.Shape kernel, T.Shape out) => (forall s. (AG.Context s) => AG.Tensor s input Double -> AG.Tensor s kernel Double -> AG.Tensor s out Double) -> ([Double] -> [Double] -> [Double]) -> H.PropertyT IO ()
checkBinary f reference = do
  let xs = take (T.shapeSize @input) $ cycle [0.2, -0.3, 0.7, 0.1, -0.5]
      ws = take (T.shapeSize @kernel) $ cycle [-0.1, 0.6, 0.4, -0.2]
      seeds = take (T.shapeSize @out) $ cycle [0.3, -0.5, 0.8, 0.2]
      (output, backward) = AG.pullback (uncurry f) (T.fromList @input xs, T.fromList @kernel ws)
      (dx, dw) = backward $ T.fromList seeds
      (actual, inputGradient, weightGradient) = T.run T.CPU (output, dx, dw)
      objective x w = sum $ zipWith (*) seeds $ reference x w
  approxList (T.toList actual) $ reference xs ws
  approxList (T.toList inputGradient) $ differences (`objective` ws) xs
  approxList (T.toList weightGradient) $ differences (objective xs) ws

propSelection :: H.Property
propSelection = H.withTests 1 $ H.property $ do
  let (_, tied) = AG.backprop (T.maxPool2d @'[2, 2] @'[1, 1] @'[0, 0, 0, 0] @'[1, 1]) $ T.ones @'[1, 1, 2, 2] @Double
      (_, overlap) = AG.backprop (T.maxPool2d @'[2, 2] @'[1, 1] @'[0, 0, 0, 0] @'[1, 1]) $ T.fromList @'[1, 1, 3, 3] [1, 2, 3, 4, 9, 5, 6, 7, 8 :: Double]
      (_, nan) = AG.backprop (T.maxPool2d @'[2, 2] @'[1, 1] @'[0, 0, 0, 0] @'[1, 1]) $ T.fromList @'[1, 1, 2, 2] [0, 0 / 0, 0 / 0, 2 :: Double]
      (a, b, c) = T.run T.CPU (tied, overlap, nan)
  T.toList a H.=== [1, 0, 0, 0]
  T.toList b H.=== [0, 0, 0, 0, 4, 0, 0, 0, 0]
  T.toList c H.=== [0, 1, 0, 0]

propPrepared :: H.Property
propPrepared = H.withTests 1 $ H.property $ do
  let execute = T.run1 T.CPU $ AG.backprop (T.maxPool2d @'[2, 2] @'[1, 1] @'[0, 0, 0, 0] @'[1, 1])
      (first, dx) = execute $ T.fromListData @'[1, 1, 2, 2] [1, 4, 2, 3 :: Double]
      (second, dx2) = execute $ T.fromListData [8, 4, 2, 3]
  T.toList first H.=== [4]
  T.toList dx H.=== [0, 1, 0, 0]
  T.toList second H.=== [8]
  T.toList dx2 H.=== [1, 0, 0, 0]

tests :: TestTree
tests =
  testGroup
    "spatial gradients"
    [ testProperty "grouped dilated conv1d gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 2, 3] @'[4, 1, 2] @'[1, 4, 3] (T.conv1d @'[1] @'[1, 1] @'[2] @2) $ R.convWith [1, 2, 3] [4, 1, 2] [1] [1, 1] [2] 2,
      testProperty "grouped dilated conv2d gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 4, 3, 3] @'[4, 2, 2, 2] @'[1, 4, 2, 3] (T.conv2d @'[1, 1] @'[1, 0, 0, 1] @'[2, 1] @2) $ R.convWith [1, 4, 3, 3] [4, 2, 2, 2] [1, 1] [1, 0, 0, 1] [2, 1] 2,
      testProperty "grouped dilated conv3d gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 2, 2, 2, 2] @'[2, 1, 1, 2, 1] @'[1, 2, 2, 2, 2] (T.conv3d @'[1, 1, 1] @'[0, 0, 1, 1, 0, 0] @'[1, 2, 1] @2) $ R.convWith [1, 2, 2, 2, 2] [2, 1, 1, 2, 1] [1, 1, 1] [0, 0, 1, 1, 0, 0] [1, 2, 1] 2,
      testProperty "grouped dilated convTranspose1d gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 2, 2] @'[2, 2, 2] @'[1, 4, 5] (T.convTranspose1d @'[2] @'[1, 0] @'[2] @2 @'[1]) $ R.transposedWith [1, 2, 2] [2, 2, 2] [2] [1, 0] [2] 2 [1],
      testProperty "grouped dilated convTranspose2d gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 2, 2, 2] @'[2, 1, 2, 2] @'[1, 2, 5, 2] (T.convTranspose2d @'[2, 1] @'[1, 0, 0, 1] @'[2, 1] @2 @'[1, 0]) $ R.transposedWith [1, 2, 2, 2] [2, 1, 2, 2] [2, 1] [1, 0, 0, 1] [2, 1] 2 [1, 0],
      testProperty "grouped dilated convTranspose3d gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 2, 1, 2, 2] @'[2, 1, 2, 1, 2] @'[1, 2, 2, 4, 5] (T.convTranspose3d @'[1, 2, 2] @'[0, 1, 0, 0, 1, 0] @'[2, 1, 2] @2 @'[0, 1, 1]) $ R.transposedWith [1, 2, 1, 2, 2] [2, 1, 2, 1, 2] [1, 2, 2] [0, 1, 0, 0, 1, 0] [2, 1, 2] 2 [0, 1, 1],
      testProperty "empty grouped gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 0, 2] @'[2, 0, 1] @'[1, 2, 2] (T.conv1d @'[1] @'[0, 0] @'[1] @2) $ R.convWith [1, 0, 2] [2, 0, 1] [1] [0, 0] [1] 2,
      testProperty "dilated max pooling gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 4, 4] @'[1, 1, 4, 4] (T.maxPool2d @'[2, 2] @'[1, 1] @'[1, 1, 1, 1] @'[2, 2]) (R.poolWith [1, 1, 4, 4] [2, 2] [1, 1] [1, 1, 1, 1] [2, 2] maximum) [-16 .. -1],
      testProperty "dilated patch extraction gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 3, 4] @'[1, 4, 6] (T.extractPatches2d @'[2, 2] @'[1, 2] @'[1, 0, 1, 1] @'[2, 1]) (R.patchesWith [1, 1, 3, 4] [2, 2] [1, 2] [1, 0, 1, 1] [2, 1]) [1 .. 12],
      testProperty "dilated patch combination gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 4, 6] @'[1, 1, 3, 4] (T.combinePatches2d @'[1, 1, 3, 4] @'[2, 2] @'[1, 2] @'[1, 0, 1, 1] @'[2, 1]) (R.combineWith [1, 1, 3, 4] [2, 2] [1, 2] [1, 0, 1, 1] [2, 1]) [1 .. 24],
      testProperty "dilated 1D patches" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 3] @'[1, 2, 2] (T.extractPatches1d @'[2] @'[1] @'[1, 0] @'[2]) (R.patchesWith [1, 1, 3] [2] [1] [1, 0] [2]) [1 .. 3],
      testProperty "dilated 3D patches" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 2, 2, 2] @'[1, 2, 8] (T.extractPatches3d @'[1, 2, 1] @'[1, 1, 1] @'[0, 0, 1, 1, 0, 0] @'[1, 2, 1]) (R.patchesWith [1, 1, 2, 2, 2] [1, 2, 1] [1, 1, 1] [0, 0, 1, 1, 0, 0] [1, 2, 1]) [1 .. 8],
      testProperty "convTranspose1d gradients with output padding" $ H.withTests 1 $ H.property $ checkBinary @'[1, 2, 3] @'[2, 1, 2] @'[1, 1, 9] (T.convTranspose1d @'[3] @'[1, 0] @'[1] @1 @'[2]) $ R.transposed [1, 2, 3] [2, 1, 2] [3] [1, 0] [2],
      testProperty "convTranspose3d gradients with output padding" $ H.withTests 1 $ H.property $ checkBinary @'[1, 1, 2, 2, 2] @'[1, 2, 2, 1, 2] @'[1, 2, 4, 2, 5] (T.convTranspose3d @'[2, 1, 3] @'[1, 0, 0, 0, 2, 0] @'[1, 1, 1] @1 @'[1, 0, 2]) $ R.transposed [1, 1, 2, 2, 2] [1, 2, 2, 1, 2] [2, 1, 3] [1, 0, 0, 0, 2, 0] [1, 0, 2],
      testProperty "conv1d gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 2, 4] @'[2, 2, 2] @'[1, 2, 2] (T.conv1d @'[2] @'[1, 0] @'[1] @1) $ R.conv [1, 2, 4] [2, 2, 2] [2] [1, 0],
      testProperty "conv3d gradients" $ H.withTests 1 $ H.property $ checkBinary @'[1, 1, 2, 2, 2] @'[2, 1, 2, 1, 2] @'[1, 2, 2, 2, 2] (T.conv3d @'[1, 2, 1] @'[1, 0, 0, 1, 1, 0] @'[1, 1, 1] @1) $ R.conv [1, 1, 2, 2, 2] [2, 1, 2, 1, 2] [1, 2, 1] [1, 0, 0, 1, 1, 0],
      testProperty "transposed convolution gradients with output padding" $ H.withTests 1 $ H.property $ checkBinary @'[1, 2, 2, 2] @'[2, 1, 2, 3] @'[1, 1, 4, 2] (T.convTranspose2d @'[2, 1] @'[1, 0, 1, 1] @'[1, 1] @1 @'[1, 0]) $ R.transposed [1, 2, 2, 2] [2, 1, 2, 3] [2, 1] [1, 0, 1, 1] [1, 0],
      testProperty "extract patches adjoint sums overlaps" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 2, 3] @'[1, 4, 6] (T.extractPatches2d @'[2, 2] @'[1, 1] @'[1, 0, 0, 1] @'[1, 1]) (R.patches [1, 1, 2, 3] [2, 2] [1, 1] [1, 0, 0, 1]) [1 .. 6],
      testProperty "combine patches adjoint extracts windows" $ H.withTests 1 $ H.property $ checkUnary @'[1, 4, 6] @'[1, 1, 2, 3] (T.combinePatches2d @'[1, 1, 2, 3] @'[2, 2] @'[1, 1] @'[1, 0, 0, 1] @'[1, 1]) (R.combine [1, 1, 2, 3] [2, 2] [1, 1] [1, 0, 0, 1]) [1 .. 24],
      testProperty "max pooling gradients exclude padding" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 3, 3] @'[1, 1, 3, 3] (T.maxPool2d @'[2, 2] @'[1, 1] @'[1, 0, 1, 0] @'[1, 1]) (R.pool [1, 1, 3, 3] [2, 2] [1, 1] [1, 0, 1, 0] maximum) [-9 .. -1],
      testProperty "average pooling gradients including padding" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 3, 3] @'[1, 1, 3, 3] (T.avgPool2d @'[2, 2] @'[1, 1] @'[1, 0, 1, 0] True) (R.pool [1, 1, 3, 3] [2, 2] [1, 1] [1, 0, 1, 0] ((/ 4) . sum)) [1 .. 9],
      testProperty "average pooling gradients excluding padding" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 3, 3] @'[1, 1, 3, 3] (T.avgPool2d @'[2, 2] @'[1, 1] @'[1, 0, 1, 0] False) (R.pool [1, 1, 3, 3] [2, 2] [1, 1] [1, 0, 1, 0] (\xs -> sum xs / fromIntegral (length xs))) [1 .. 9],
      testProperty "maxPool1d gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 4] @'[1, 1, 2] (T.maxPool1d @'[3] @'[2] @'[1, 0] @'[1]) (R.pool [1, 1, 4] [3] [2] [1, 0] maximum) [-4 .. -1],
      testProperty "avgPool1d gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 4] @'[1, 1, 2] (T.avgPool1d @'[3] @'[2] @'[1, 0] True) (R.pool [1, 1, 4] [3] [2] [1, 0] ((/ 3) . sum)) [1 .. 4],
      testProperty "maxPool3d gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 2, 2, 2] @'[1, 1, 2, 2, 2] (T.maxPool3d @'[2, 2, 2] @'[1, 1, 1] @'[1, 0, 1, 0, 1, 0] @'[1, 1, 1]) (R.pool [1, 1, 2, 2, 2] [2, 2, 2] [1, 1, 1] [1, 0, 1, 0, 1, 0] maximum) [-8 .. -1],
      testProperty "avgPool3d gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 2, 2, 2] @'[1, 1, 2, 2, 2] (T.avgPool3d @'[2, 2, 2] @'[1, 1, 1] @'[1, 0, 1, 0, 1, 0] False) (R.pool [1, 1, 2, 2, 2] [2, 2, 2] [1, 1, 1] [1, 0, 1, 0, 1, 0] (\xs -> sum xs / fromIntegral (length xs))) [1 .. 8],
      testProperty "extractPatches1d gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 3] @'[1, 2, 3] (T.extractPatches1d @'[2] @'[1] @'[1, 0] @'[1]) (R.patches [1, 1, 3] [2] [1] [1, 0]) [1 .. 3],
      testProperty "combinePatches1d gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 2, 3] @'[1, 1, 3] (T.combinePatches1d @'[1, 1, 3] @'[2] @'[1] @'[1, 0] @'[1]) (R.combine [1, 1, 3] [2] [1] [1, 0]) [1 .. 6],
      testProperty "extractPatches3d gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 1, 2, 2, 2] @'[1, 4, 2] (T.extractPatches3d @'[2, 1, 2] @'[1, 1, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1]) (R.patches [1, 1, 2, 2, 2] [2, 1, 2] [1, 1, 1] [0, 0, 0, 0, 0, 0]) [1 .. 8],
      testProperty "combinePatches3d gradients" $ H.withTests 1 $ H.property $ checkUnary @'[1, 4, 2] @'[1, 1, 2, 2, 2] (T.combinePatches3d @'[1, 1, 2, 2, 2] @'[2, 1, 2] @'[1, 1, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1]) (R.combine [1, 1, 2, 2, 2] [2, 1, 2] [1, 1, 1] [0, 0, 0, 0, 0, 0]) [1 .. 8],
      testProperty "max pooling selects first ties and accumulates overlaps" propSelection,
      testProperty "prepared pooling uses the current input's maxima" propPrepared
    ]
