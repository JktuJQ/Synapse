-- | Tests @Synapse.NN@ module and its submodules.


module NNTest
    ( tests
    )
    where


import Synapse.Tensors
import qualified Synapse.Tensors.Vec as V

import Synapse.NN.Models
import Synapse.NN.Layers
import Synapse.NN.Optimizers
import Synapse.NN.LearningRates
import Synapse.NN.Losses
import Synapse.NN.Batching
import Synapse.NN.Training

import System.Random

import Graphics.EasyPlot

import Test.HUnit


testSin :: Test  -- -3 sin (x + 5)
testSin = TestLabel "testSin" $ TestCase $ do
    let sinFn x = (-3.0) * sin (x + 5.0)
    let model = buildSequentialModel (InputSize 1) [ Layer . layerDense 1
                                                   , Layer . layerActivation cos
                                                   , Layer . layerDense 1
                                                   ] :: SequentialModel Double

    let dataset = Dataset $ V.fromList $ [Sample (singleton x) (sinFn $ singleton x) | x <- [-pi, -pi+0.2 .. pi]]
    (trainedModel, losses, _) <- train model
                                       (SGD 0.2 False)
                                       (Hyperparameters 500 16 dataset (LearningRate $ const 0.01) (Loss mse) V.empty)
                                       emptyCallbacks
                                       (mkStdGen 1)
    _ <- plot (PNG "test/plots/sin.png")
              [ Data2D [Title "predicted sin", Style Lines, Color Red] [Range (-pi) pi] [(x, unSingleton $ forward (singleton x) trainedModel) | x <- [-pi, -pi+0.05 .. pi]]
              , Data2D [Title "true sin", Style Lines, Color Green] [Range (-pi) pi] [(x, sinFn x) | x <- [-pi, -pi+0.05 .. pi]]
              ]

    let unpackedLosses = unRecordedMetric (unsafeIndex losses 0)
    let lastLoss = unsafeIndex unpackedLosses (V.size unpackedLosses - 1)

    assertBool "trained well enough" (lastLoss < 0.01)

testSqrt :: Test  -- sqrt(x)
testSqrt = TestLabel "testSqrt" $ TestCase $ do
    let sqrtFn x = sqrt x
    let model = buildSequentialModel (InputSize 1) [ Layer . layerDense 1
                                                   , Layer . layerActivation tanh
                                                   , Layer . layerDense 1
                                                   ] :: SequentialModel Double

    let dataset = Dataset $ V.fromList $ [Sample (singleton x) (sqrtFn $ singleton x) | x <- [0.0, 0.2 .. 4.0]]
    (trainedModel, losses, _) <- train model
                                       (SGD 0.2 True)
                                       (Hyperparameters 500 16 dataset (LearningRate $ const 0.01) (Loss mse) V.empty)
                                       emptyCallbacks
                                       (mkStdGen 1)
    _ <- plot (PNG "test/plots/sqrt.png")
               [ Data2D [Title "predicted sqrt", Style Lines, Color Red] [Range 0.0 6.0] $ [(x, unSingleton $ forward (singleton x) trainedModel) | x <- [0.0, 0.05 .. 4.0]]
               , Data2D [Title "true sqrt", Style Lines, Color Green] [Range 0.0 6.0] $ [(x, sqrtFn x) | x <- [0.0, 0.05 .. 4.0]]
               ]

    let unpackedLosses = unRecordedMetric (unsafeIndex losses 0)
    let lastLoss = unsafeIndex unpackedLosses (V.size unpackedLosses - 1)

    assertBool "trained well enough" (lastLoss < 0.01)

testTrigonometry :: Test  -- sin(2.0 * cos(x) + 3.0) + 2.5
testTrigonometry = TestLabel "testTrigonometry" $ TestCase $ do
    let trigonometryFn x = sin (2.0 * cos x + 3.0) + 2.5
    let model = buildSequentialModel (InputSize 1) [ Layer . layerDense 1
                                                   , Layer . layerActivation sin
                                                   , Layer . layerDense 1
                                                   , Layer . layerActivation sin
                                                   , Layer . layerDense 1
                                                   ] :: SequentialModel Double

    let dataset = Dataset $ V.fromList $ [Sample (singleton x) (trigonometryFn $ singleton x) | x <- [-(2.0 * pi),((-(2.0 * pi)) + 0.1)..(2.0 * pi)]]
                                         
    (trainedModel, losses, _) <- train model
                                       (SGD 0.3 True)
                                       (Hyperparameters 1000 1 dataset (LearningRate $ const 0.001) (Loss mse) V.empty)
                                       emptyCallbacks
                                       (mkStdGen 1)

    _ <- plot (PNG "test/plots/trigonometry.png")
               [ Data2D [Title "predicted trigonometry", Style Lines, Color Red] [Range ((-2.0) * pi) (2.0 * pi)] $ [(x, unSingleton $ forward (singleton x) trainedModel) | x <- [-(2.0 * pi),((-(2.0 * pi)) + 0.1)..(2.0 * pi)]]
               , Data2D [Title "true trigonometry", Style Lines, Color Green] [Range ((-2.0) * pi) (2.0 * pi)] $ [(x, trigonometryFn x) | x <- [-(2.0 * pi),((-(2.0 * pi)) + 0.1)..(2.0 * pi)]]
               ]

    let unpackedLosses = unRecordedMetric (unsafeIndex losses 0)
    let lastLoss = unsafeIndex unpackedLosses (V.size unpackedLosses - 1)

    assertBool "trained well enough" (lastLoss < 0.01)


tests :: Test
tests = TestLabel "NNTest" $ TestList
    [ testSin
    , testSqrt
    , testTrigonometry
    ]

