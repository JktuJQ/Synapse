{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Dense (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import System.Random (mkStdGen64)
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

parameters :: NN.Dense 2 3 T.Tensor Double
parameters = NN.Dense (T.fromList @'[2, 3] [1, 2, 3, 4, 5, 6]) (T.fromList @'[3] [0.5, -1, 2])

tests :: TestTree
tests =
  testGroup
    "dense"
    [ testProperty "dense maps a feature vector to the statically determined output" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [1, 2 :: Double]
            output :: T.Tensor '[3] Double
            output = NN.dense parameters input
        values output H.=== [9.5, 11, 17],
      testProperty "dense shares weights and broadcasts bias across a batch" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            output :: T.Tensor '[2, 3] Double
            output = NN.dense parameters input
        values output H.=== [9.5, 11, 17, 19.5, 25, 35],
      testProperty "dense preserves multiple leading dimensions" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 1, 2] [1, 2, 3, 4 :: Double]
            output :: T.Tensor '[2, 1, 3] Double
            output = NN.dense parameters input
            higherRank :: T.Tensor '[1, 2, 1, 3] Double
            higherRank = NN.dense parameters (T.fromList @'[1, 2, 1, 2] [1, 2, 3, 4])
        values output H.=== [9.5, 11, 17, 19.5, 25, 35]
        values higherRank H.=== values output,
      testProperty "dense accepts empty batches and zero output widths" $ H.withTests 1 $ H.property $ do
        values (NN.dense parameters (T.zeros @'[0, 2])) H.=== []
        values (NN.dense parameters (T.zeros @'[2, 0, 2])) H.=== []
        let zeroOutput = NN.Dense (T.zeros @'[2, 0]) (T.zeros @'[0]) :: NN.Dense 2 0 T.Tensor Double
        values (NN.dense zeroOutput (T.ones @'[3, 2])) H.=== [],
      testProperty "zero input widths produce the broadcast bias" $ H.withTests 1 $ H.property $ do
        let zeroInput = NN.Dense (T.zeros @'[0, 3]) (T.fromList @'[3] [0.5, -1, 2]) :: NN.Dense 0 3 T.Tensor Double
        values (NN.dense zeroInput (T.zeros @'[0])) H.=== [0.5, -1, 2]
        values (NN.dense zeroInput (T.zeros @'[2, 0])) H.=== [0.5, -1, 2, 0.5, -1, 2],
      testProperty "dense parameters and inputs differentiate through the same function" $ H.withTests 1 $ H.property $ do
        let layer = NN.Dense (T.fromList @'[2, 2] [1, 2, 3, 4]) (T.fromList @'[2] [0.5, -1]) :: NN.Dense 2 2 T.Tensor Double
            input = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            (loss, (gradient, dx)) = AG.backprop (\(p, x) -> T.sum (NN.dense p x)) (layer, input)
        values loss H.=== [53]
        values (NN.denseWeights gradient) H.=== [4, 4, 6, 6]
        values (NN.denseBias gradient) H.=== [2, 2]
        values dx H.=== [3, 7, 3, 7],
      testProperty "dense bias gradients include every leading batch dimension" $ H.withTests 1 $ H.property $ do
        let input = T.ones @'[2, 4, 2] @Double
            (_, gradient) = AG.backprop (\p -> T.sum (NN.dense p (AG.constant input))) parameters
        values (NN.denseWeights gradient) H.=== replicate 6 8
        values (NN.denseBias gradient) H.=== replicate 3 8,
      testProperty "evaluate agrees with inference and activations compose as ordinary functions" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [-1, 0 :: Double]
            inferred = NN.relu (NN.dense parameters input)
            evaluated = AG.evaluate (\p -> NN.relu (NN.dense p (AG.constant input))) parameters
        values inferred H.=== [0, 0, 0]
        values evaluated H.=== values inferred,
      testProperty "dense supports externally initialized single-precision parameters" $ H.withTests 1 $ H.property $ do
        let layer = NN.Dense (NN.xavierNormal @'[2, 3] 2 3 1 (mkStdGen64 2026)) (T.zeros @'[3]) :: NN.Dense 2 3 T.Tensor Float
            output = T.toList (T.run T.CPU (NN.dense layer (T.ones @'[2])))
        length output H.=== 3
        H.assert (all (\x -> not (isNaN x || isInfinite x)) output),
      testProperty "prepared dense inference accepts changing parameters" $ H.withTests 1 $ H.property $ do
        let execute = T.run3 T.CPU $ \batch layerWeights layerBias -> NN.dense (NN.Dense layerWeights layerBias) batch
            input = T.run T.CPU (T.fromList @'[2] [1, 2 :: Double])
            w = T.run T.CPU (NN.denseWeights parameters)
            b = T.run T.CPU (NN.denseBias parameters)
            zeroBias = T.run T.CPU (T.zeros @'[3] @Double)
        T.toList (execute input w b) H.=== [9.5, 11, 17]
        T.toList (execute input w zeroBias) H.=== [9, 12, 15],
      testProperty "prepared dense gradients keep the record registration separate from execution outputs" $ H.withTests 1 $ H.property $ do
        let execute = T.run3 T.CPU $ \batch layerWeights layerBias ->
              let (objective, gradient) = AG.backprop (\p -> T.sum (NN.dense p (AG.constant batch))) (NN.Dense layerWeights layerBias)
               in (objective, NN.denseWeights gradient, NN.denseBias gradient)
            input = T.run T.CPU (T.fromList @'[2] [1, 2 :: Double])
            (loss, dw, db) = execute input (T.run T.CPU (NN.denseWeights parameters)) (T.run T.CPU (NN.denseBias parameters))
        T.toList loss H.=== [37.5]
        T.toList dw H.=== [1, 1, 1, 2, 2, 2]
        T.toList db H.=== [1, 1, 1],
      testProperty "stopped dense parameters still allow differentiation with respect to inputs" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [1, 2 :: Double]
            (_, gradient) = AG.backprop (T.sum . NN.dense (AG.constant parameters)) input
        values gradient H.=== [6, 15],
      testProperty "dense mse gradients agree with the analytic weight and bias derivatives" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [0.7, 1.2 :: Double]
            target = T.fromList @'[3] [1, 2, 3 :: Double]
            (_, gradient) = AG.backprop (\p -> NN.mse T.mean (NN.dense p (AG.constant input)) (AG.constant target)) parameters
            prediction = [6, 6.4, 11.3 :: Double]
            residualGradient = zipWith (\p t -> 2 * (p - t) / 3) prediction [1, 2, 3]
        approxList (values (NN.denseWeights gradient)) (map (* 0.7) residualGradient ++ map (* 1.2) residualGradient)
        approxList (values (NN.denseBias gradient)) residualGradient,
      testProperty "dense rejects an incorrect feature count even for empty batches" $
        H.withTests 1 $
          Compiler.rejects "test/NNTest/Fixtures/Rejected/DenseFeatures.hs" "Matmul inner dimensions must match",
      testProperty "dense rejects scalar inputs" $
        H.withTests 1 $
          Compiler.rejects "test/NNTest/Fixtures/Rejected/DenseScalar.hs" "Matmul requires tensors of rank at least one",
      testProperty "dense rejects an incorrect declared output shape" $
        H.withTests 1 $
          Compiler.rejects "test/NNTest/Fixtures/Rejected/DenseOutput.hs" "Cannot broadcast dimension 3 to 4"
    ]
