{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Layers.Normalization (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

finiteDifference :: ([Double] -> Double) -> [Double] -> [Double]
finiteDifference objective input =
  let shift index delta = zipWith (\i x -> if i == index then x + delta else x) [0 :: Int ..] input
   in [(objective (shift i 1e-5) - objective (shift i (-1e-5))) / 2e-5 | i <- [0 .. length input - 1]]

tests :: TestTree
tests =
  testGroup
    "normalization"
    [ testProperty "LayerNorm rejects a mismatched normalized suffix" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/LayerNormSuffix.hs" "LayerNorm",
      testProperty "BatchNorm rejects a mismatched channel count" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/BatchNormChannels.hs" "Couldn't match type",
      testProperty "LayerNorm applies per-element affine parameters after trailing normalization" $ H.withTests 1 $ H.property $ do
        let x = T.fromList @'[2, 3] [1, 2, 3, 4, 6, 8 :: Double]
            p = NN.LayerNorm (T.fromList @'[3] [1, 2, 3]) (T.fromList @'[3] [0.5, 1, 1.5])
            normalized = NN.layerNorm 1 p x
            row scale = [-scale + 0.5, 1, 3 * scale + 1.5]
        approxList (values normalized) (row (1 / sqrt (5 / 3)) ++ row (2 / sqrt (11 / 3))),
      testProperty "LayerNorm input and affine gradients match independent references" $ H.withTests 1 $ H.property $ do
        let xs = [1, 2, 4 :: Double]
            x = T.fromList @'[1, 3] xs
            p = NN.LayerNorm (T.fromList @'[3] [1, 2, 3]) (T.zeros @'[3])
            seed = T.fromList @'[1, 3] [2, 3, 5]
            (_, backward) = AG.pullback (\(input, parameters) -> NN.layerNorm 0.1 parameters input) (x, p)
            (dx, dp) = backward seed
            reference row =
              let rowMean = sum row / 3
                  rowVariance = sum [(v - rowMean) ^ (2 :: Int) | v <- row] / 3
               in sum $ zipWith (*) [2, 6, 15] [(v - rowMean) / sqrt (rowVariance + 0.1) | v <- row]
            mean = sum xs / 3
            variance = sum [(v - mean) ^ (2 :: Int) | v <- xs] / 3
        approxList (values dx) (finiteDifference reference xs)
        approxList (values $ NN.layerNormWeights dp) (zipWith (*) [2, 3, 5] [(v - mean) / sqrt (variance + 0.1) | v <- xs])
        approxList (values $ NN.layerNormBias dp) [2, 3, 5],
      testProperty "LayerNorm supports multiple trailing dimensions and Float" $ H.withTests 1 $ H.property $ do
        let x = T.fromList @'[2, 2, 2] [1, 2, 3, 4, 5, 6, 7, 8 :: Float]
            p = NN.LayerNorm (T.full @'[2, 2] 1) (T.zeros @'[2, 2])
            result = NN.layerNorm 0.75 p x
        approxList (map realToFrac (T.toList $ T.run T.CPU result) :: [Double]) (concat $ replicate 2 $ map (/ sqrt 2) [-1.5, -0.5, 0.5, 1.5]),
      testProperty "BatchNorm training uses population variance and updates unbiased running variance" $ H.withTests 1 $ H.property $ do
        let p = NN.BatchNorm (T.full @'[2] (1 :: Double)) 0
            x = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            (output, state) = NN.batchNorm @1 NN.Train 1 0.1 p NN.initialBatchNormState x
        approxList (values output) (map (/ sqrt 2) [-1, -1, 1, 1])
        approxList (values $ NN.batchNormMean state) [0.2, 0.3]
        approxList (values $ NN.batchNormVariance state) [1.1, 1.1],
      testProperty "BatchNorm eval reuses state and does not inspect momentum" $ H.withTests 1 $ H.property $ do
        let p = NN.BatchNorm (T.fromList @'[2] [2, 3 :: Double]) (T.fromList @'[2] [0.5, -0.5])
            state = NN.BatchNormState (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [3, 8])
            x = T.fromList @'[1, 2] [1, 2]
            (output, next) = NN.batchNorm @1 NN.Eval 1 undefined p state x
        approxList (values output) [-0.5, -1.5]
        values (NN.batchNormMean next) H.=== [2, 3]
        values (NN.batchNormVariance next) H.=== [3, 8],
      testProperty "BatchNorm differentiates batch statistics and affine parameters" $ H.withTests 1 $ H.property $ do
        let xs = [1, 2, 3, 5 :: Double]
            x = T.fromList @'[2, 2] xs
            p = NN.BatchNorm (T.full @'[2] 1) 0
            (_, backward) = AG.pullback (\(input, parameters) -> fst $ NN.batchNorm @1 NN.Train 0.1 0.1 parameters NN.initialBatchNormState input) (x, p)
            (dx, dp) = backward (T.fromList @'[2, 2] [1, 2, 4, 8])
            reference [a, b, c, d] =
              let first = (a - c) / 2 / sqrt (((a - c) / 2) ^ (2 :: Int) + 0.1)
                  second = (b - d) / 2 / sqrt (((b - d) / 2) ^ (2 :: Int) + 0.1)
               in (-3) * first + (-6) * second
            reference _ = error "invalid reference input"
        approxList (values dx) (finiteDifference reference xs)
        approxList (values $ NN.batchNormWeights dp) [3 / sqrt 1.1, 9 / sqrt 2.35]
        approxList (values $ NN.batchNormBias dp) [5, 10],
      testProperty "BatchNorm state and auxiliary autograd output pass through a prepared training step" $ H.withTests 1 $ H.property $ do
        let execute = T.run3 T.CPU $ \parametersInput running features ->
              AG.backprop
                ( \parameters ->
                    let (output, next) = NN.batchNorm @1 NN.Train 1 0.5 parameters running (AG.constant features)
                     in (T.sum (output * output), AG.constant next)
                )
                parametersInput
            p = T.run T.CPU $ NN.BatchNorm (T.full @'[1] (1 :: Double)) 0
            state = T.run T.CPU $ NN.initialBatchNormState @1 @Double
            x = T.fromListData @'[2, 1] [1, 3 :: Double]
            ((loss, first), gradients) = execute p state x
            ((_, second), _) = execute p first x
        approxList (T.toList loss) [1]
        approxList (T.toList $ NN.batchNormWeights gradients) [2]
        approxList (T.toList $ NN.batchNormBias gradients) [0]
        approxList (T.toList $ NN.batchNormMean first) [1]
        approxList (T.toList $ NN.batchNormMean second) [1.5]
        approxList (T.toList $ NN.batchNormVariance second) [1.75],
      testProperty "BatchNorm handles spatial axes and empty training batches" $ H.withTests 1 $ H.property $ do
        let p = NN.BatchNorm (T.full @'[2] (1 :: Double)) 0
            x = T.fromList @'[2, 1, 2] [1, 3, 10, 14 :: Double]
            (output, _) = NN.batchNorm @0 NN.Train 1 0.1 p NN.initialBatchNormState x
            empty = T.zeros @'[0, 2] @Double
            (result, state) = NN.batchNorm @1 NN.Train 1 undefined p NN.initialBatchNormState empty
        approxList (values output) [-(1 / sqrt 2), 1 / sqrt 2, -(2 / sqrt 5), 2 / sqrt 5]
        values result H.=== []
        values (NN.batchNormMean state) H.=== [0, 0]
        values (NN.batchNormVariance state) H.=== [1, 1]
    ]
