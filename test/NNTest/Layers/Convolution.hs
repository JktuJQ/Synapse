{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Layers.Convolution (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.NN.Layers as Layers
import qualified Synapse.NN.Layers.Convolution as Conv
import qualified Synapse.NN.Weights as Weights
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import qualified TensorTest.Ops.Spatial.Reference as R
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

addBias :: Int -> Int -> Int -> [Double] -> [Double] -> [Double]
addBias batch channels spatial biases output = zipWith (+) output [biases !! channel | _ <- [1 .. batch], channel <- [0 .. channels - 1], _ <- [1 .. spatial]]

propForward :: H.Property
propForward = H.withTests 1 $ H.property $ do
  let parameters = Layers.Convolution (T.fromList @'[2, 1, 1] [2, 3]) (T.fromList @'[2] [0.5, -1]) :: Layers.Convolution 1 2 '[1] 1 T.Tensor Double
      result = Layers.conv1d @'[1] @'[0, 0] @'[1] parameters $ T.fromList @'[1, 1, 3] [1, 2, 3]
      grouped = NN.Convolution (T.fromList @'[4, 1, 2, 2] [1 .. 16]) (T.fromList @'[4] [0.1, 0.2, 0.3, 0.4]) :: NN.Convolution 2 4 '[2, 2] 2 T.Tensor Double
      result2 = NN.conv2d @'[1, 2] @'[1, 0, 0, 1] @'[1, 1] grouped $ T.fromList @'[2, 2, 3, 4] [1 .. 48]
      volume = Conv.Convolution (T.fromList @'[2, 1, 1, 2, 1] [1 .. 4]) (T.fromList @'[2] [1, -2]) :: Conv.Convolution 1 2 '[1, 2, 1] 1 T.Tensor Double
      result3 = Conv.conv3d @'[1, 1, 1] @'[0, 0, 1, 1, 0, 0] @'[1, 2, 1] volume $ T.fromList @'[1, 1, 2, 3, 2] [1 .. 12]
  values result H.=== [2.5, 4.5, 6.5, 2, 5, 8]
  approxList (values result2) $ addBias 2 4 6 [0.1, 0.2, 0.3, 0.4] $ R.convWith [2, 2, 3, 4] [4, 1, 2, 2] [1, 2] [1, 0, 0, 1] [1, 1] 2 [1 .. 48] [1 .. 16]
  approxList (values result3) $ addBias 1 2 12 [1, -2] $ R.convWith [1, 1, 2, 3, 2] [2, 1, 1, 2, 1] [1, 1, 1] [0, 0, 1, 1, 0, 0] [1, 2, 1] 1 [1 .. 12] [1 .. 4]

propTranspose :: H.Property
propTranspose = H.withTests 1 $ H.property $ do
  let parameters = NN.TransposedConvolution (T.fromList @'[2, 2, 2] [1 .. 8]) (T.fromList @'[4] [1, 2, 3, 4]) :: NN.TransposedConvolution 2 4 '[2] 2 T.Tensor Double
      result = NN.convTranspose1d @'[2] @'[1, 0] @'[2] @'[1] parameters $ T.fromList @'[1, 2, 2] [1 .. 4]
      plane = NN.TransposedConvolution (T.fromList @'[1, 2, 1, 1] [2, 3]) (T.fromList @'[2] [0.5, -1]) :: NN.TransposedConvolution 1 2 '[1, 1] 1 T.Tensor Double
      result2 = NN.convTranspose2d @'[2, 2] @'[0, 0, 0, 0] @'[1, 1] @'[1, 1] plane $ T.fromList @'[1, 1, 2, 2] [1 .. 4]
      volume = NN.TransposedConvolution (T.fromList @'[2, 1, 2, 1, 2] [1 .. 8]) (T.fromList @'[2] [1, -1]) :: NN.TransposedConvolution 2 2 '[2, 1, 2] 2 T.Tensor Double
      result3 = NN.convTranspose3d @'[1, 2, 2] @'[0, 1, 0, 0, 1, 0] @'[2, 1, 2] @'[0, 1, 1] volume $ T.fromList @'[1, 2, 1, 2, 2] [1 .. 8]
  approxList (values result) $ addBias 1 4 5 [1, 2, 3, 4] $ R.transposedWith [1, 2, 2] [2, 2, 2] [2] [1, 0] [2] 2 [1] [1 .. 4] [1 .. 8]
  approxList (values result2) $ addBias 1 2 16 [0.5, -1] $ R.transposedWith [1, 1, 2, 2] [1, 2, 1, 1] [2, 2] [0, 0, 0, 0] [1, 1] 1 [1, 1] [1 .. 4] [2, 3]
  approxList (values result3) $ addBias 1 2 40 [1, -1] $ R.transposedWith [1, 2, 1, 2, 2] [2, 1, 2, 1, 2] [1, 2, 2] [0, 1, 0, 0, 1, 0] [2, 1, 2] 2 [0, 1, 1] [1 .. 8] [1 .. 8]

propGradients :: H.Property
propGradients = H.withTests 1 $ H.property $ do
  let parameters = NN.Convolution (T.fromList @'[2, 1, 1] [2, 3]) (T.fromList @'[2] [0.5, -1]) :: NN.Convolution 1 2 '[1] 1 T.Tensor Double
      input = T.fromList @'[1, 1, 3] [1, 2, 3 :: Double]
      (loss, (gradient, dx)) = AG.backprop (\(p, x) -> T.sum $ NN.conv1d @'[1] @'[0, 0] @'[1] p x) (parameters, input)
  values loss H.=== [28.5]
  values (NN.convolutionWeights gradient) H.=== [6, 6]
  values (NN.convolutionBias gradient) H.=== [3, 3]
  values dx H.=== [5, 5, 5]
  let transpose = NN.TransposedConvolution (T.fromList @'[1, 2, 1, 1] [2, 3]) (T.fromList @'[2] [0.5, -1]) :: NN.TransposedConvolution 1 2 '[1, 1] 1 T.Tensor Double
      (_, (transposedGradient, transposedDX)) = AG.backprop (\(p, x) -> T.sum $ NN.convTranspose2d @'[2, 2] @'[0, 0, 0, 0] @'[1, 1] @'[1, 1] p x) (transpose, T.fromList @'[1, 1, 2, 2] [1 .. 4])
  values (NN.transposedWeights transposedGradient) H.=== [10, 10]
  values (NN.transposedBias transposedGradient) H.=== [16, 16]
  values transposedDX H.=== replicate 4 5

propRecords :: H.Property
propRecords = H.withTests 1 $ H.property $ do
  let parameters = NN.Convolution (T.fromList @'[2, 1, 1] [2, 3]) (T.fromList @'[2] [0.5, -1]) :: NN.Convolution 1 2 '[1] 1 T.Tensor Double
      execute = T.run2 T.CPU $ \p batch ->
        let (objective, gradient) = AG.backprop (\layer -> T.sum $ NN.conv1d @'[1] @'[0, 0] @'[1] layer $ AG.constant batch) p
            (updatedParameters, _) = NN.optimizerStep (NN.SGD 0.1) p gradient ()
         in (objective, updatedParameters)
      materialized = T.run T.CPU parameters
      input = T.fromListData @'[1, 1, 3] [1, 2, 3 :: Double]
      (loss, updated) = execute materialized input
  T.toList loss H.=== [28.5]
  approxList (T.toList $ NN.convolutionWeights updated) [1.4, 2.4]
  approxList (T.toList $ NN.convolutionBias updated) [0.2, -1.3]
  let dictionary = NN.extractWeights materialized
      destination = NN.Convolution (T.zeros @'[2, 1, 1]) (T.zeros @'[2]) :: NN.Convolution 1 2 '[1] 1 T.Tensor Double
  map fst (Weights.weightsToList dictionary) H.=== [["convolutionBias"], ["convolutionWeights"]]
  restored <- H.evalEither $ NN.loadWeights dictionary $ T.run T.CPU destination
  T.toList (NN.convolutionWeights restored) H.=== [2, 3]
  T.toList (NN.convolutionBias restored) H.=== [0.5, -1]

propEmpty :: H.Property
propEmpty = H.withTests 1 $ H.property $ do
  let parameters = NN.Convolution (T.ones @'[2, 1, 1]) (T.fromList @'[2] [1, 2]) :: NN.Convolution 1 2 '[1] 1 T.Tensor Double
  values (NN.conv1d @'[1] @'[0, 0] @'[1] parameters $ T.zeros @'[0, 1, 3]) H.=== []
  let noInput = NN.Convolution (T.zeros @'[2, 0, 1]) (T.fromList @'[2] [1, 2]) :: NN.Convolution 0 2 '[1] 2 T.Tensor Double
  values (NN.conv1d @'[1] @'[0, 0] @'[1] noInput $ T.zeros @'[1, 0, 2]) H.=== [1, 1, 2, 2]
  let float = NN.Convolution (T.ones @'[1, 1, 1]) (T.fromList @'[1] [0.5]) :: NN.Convolution 1 1 '[1] 1 T.Tensor Float
  approxList (T.toList $ T.run T.CPU $ NN.conv1d @'[1] @'[0, 0] @'[1] float $ T.ones @'[1, 1, 2]) [1.5, 1.5]

tests :: TestTree
tests =
  testGroup
    "convolution"
    [ testProperty "convolution layers add channel bias in 1D/2D/3D" propForward,
      testProperty "transposed layers add bias across output padding" propTranspose,
      testProperty "parameters, bias and input gradients" propGradients,
      testProperty "prepared execution, SGD and weight serialization preserve layer records" propRecords,
      testProperty "empty batches, zero input channels and Float" propEmpty,
      testProperty "convolution rejects an incorrect input channel count" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/ConvFeatures.hs" "Couldn't match type",
      testProperty "convolution rejects an incorrect kernel rank" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/ConvKernel.hs" "Conv2d requires",
      testProperty "transposed convolution validates declared output channel divisibility" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/ConvTransposeGroups.hs" "Couldn't match type"
    ]
