{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Layers.Dropout (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import System.Random (mkStdGen)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "dropout"
    [ testProperty "prepared dropout accepts changing external masks" $ H.withTests 1 $ H.property $ do
        let apply :: T.Tensor '[2] Double -> T.Tensor '[2] Bool -> T.Tensor '[2] Double
            apply input mask = NN.dropoutWithMask NN.Train 0.5 mask input
            execute = T.run2 T.CPU apply
            x = T.fromListData @'[2] [2, 4 :: Double]
            first = T.fromListData @'[2] [True, False]
            second = T.fromListData @'[2] [False, True]
        T.toList (execute x first) H.=== [4, 0]
        T.toList (execute x second) H.=== [0, 8],
      testProperty "dropout applies an external mask and differentiates only retained elements" $ H.withTests 1 $ H.property $ do
        let x = T.fromList @'[4] [1, 2, 3, 4 :: Double]
            mask = T.fromList @'[4] [True, False, True, False]
            (output, gradient) = AG.backprop (NN.dropoutWithMask NN.Train 0.5 mask) x
        values output H.=== [2, 0, 6, 0]
        values gradient H.=== [2, 0, 2, 0],
      testProperty "dropout evaluation and endpoints do not inspect unused arguments" $ H.withTests 1 $ H.property $ do
        let x = T.fromList @'[2] [2, 4 :: Double]
        values (NN.dropoutWithMask NN.Eval undefined undefined x) H.=== [2, 4]
        values (NN.dropoutWithMask NN.Train 0 undefined x) H.=== [2, 4]
        values (NN.dropoutWithMask NN.Train 1 undefined x) H.=== [0, 0]
        values (AG.gradient (NN.dropoutWithMask NN.Train 1 undefined) x) H.=== [0, 0],
      testProperty "prepared dropout samples a fresh mask from each returned state" $ H.withTests 1 $ H.property $ do
        let (initial, _) = NN.initializeDropout $ mkStdGen 42
            execute = T.run2 T.CPU $ \running features -> NN.runState (NN.dropout NN.Train 0.5 (features :: T.Tensor '[64] Double)) running
            input = T.fromListData @'[64] $ replicate 64 1
            state = T.run T.CPU initial
            (first, next) = execute state input
            (second, _) = execute next input
            (repeated, _) = execute state input
        T.toList first H.=== T.toList repeated
        H.assert $ T.toList first /= T.toList second
        H.assert $ all (`elem` [0, 2]) $ T.toList first,
      testProperty "stateful dropout gradients use exactly the sampled mask" $ H.withTests 1 $ H.property $ do
        let state = NN.DropoutState $ T.generatorFromSeed 123
            ((output, next), gradients) =
              AG.backprop
                (\input -> let (result, updated) = NN.runState (NN.dropout NN.Train 0.5 input) state in (result, AG.constant updated))
                (T.fromList @'[4] [1, 2, 3, 4 :: Double])
            (mask, expectedState) = T.bernoulli @'[4] 0.5 $ NN.dropoutGenerator state
            (actualNext, _) = T.randomWords @'[4] $ NN.dropoutGenerator next
            (expectedNext, _) = T.randomWords @'[4] expectedState
            retained = T.toList $ T.run T.CPU mask
        values output H.=== zipWith (\keep x -> if keep then x * 2 else 0) retained [1, 2, 3, 4]
        values gradients H.=== map (\keep -> if keep then 2 else 0) retained
        T.toList (T.run T.CPU actualNext) H.=== T.toList (T.run T.CPU expectedNext),
      testProperty "evaluation and endpoint probabilities preserve state" $ H.withTests 1 $ H.property $ do
        let state = NN.DropoutState $ T.generatorFromSeed 42
            input = T.fromList @'[2] [2, 4 :: Double]
            (expected, _) = T.randomWords @'[4] $ NN.dropoutGenerator state
        values (NN.evalState (NN.dropout NN.Eval undefined input) undefined) H.=== [2, 4]
        values (NN.evalState (NN.dropout NN.Train 0 input) undefined) H.=== [2, 4]
        values (NN.evalState (NN.dropout NN.Train 1 input) undefined) H.=== [0, 0]
        mapM_
          ( \(mode, probability) -> do
              let (_, next) = NN.runState (NN.dropout mode probability input) state
                  (actual, _) = T.randomWords @'[4] $ NN.dropoutGenerator next
              T.toList (T.run T.CPU actual) H.=== T.toList (T.run T.CPU expected)
          )
          [(NN.Eval, 0.5), (NN.Train, 0), (NN.Train, 1)],
      testProperty "empty dropout does not advance the generator and supports Float" $ H.withTests 1 $ H.property $ do
        let state = NN.DropoutState $ T.generatorFromSeed 42
            (empty, next) = NN.runState (NN.dropout NN.Train 0.5 $ T.zeros @'[0, 2] @Float) state
            (actual, _) = T.randomWords @'[4] $ NN.dropoutGenerator next
            (expected, _) = T.randomWords @'[4] $ NN.dropoutGenerator state
            (output, _) = NN.runState (NN.dropout NN.Train 0.5 $ T.ones @'[16] @Float) state
        T.toList (T.run T.CPU empty) H.=== []
        T.toList (T.run T.CPU actual) H.=== T.toList (T.run T.CPU expected)
        H.assert $ all (`elem` [0, 2]) $ T.toList $ T.run T.CPU output
    ]
