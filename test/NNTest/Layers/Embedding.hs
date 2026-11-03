{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Layers.Embedding (tests) where

import Control.Monad (foldM)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.NN.Weights as Weights
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

parameters :: NN.Embedding 4 2 T.Tensor Double
parameters = NN.Embedding $ T.fromList [1, 2, 3, 4, 5, 6, 7, 8]

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

seedGradient :: T.TensorData '[6] Int -> T.TensorData '[6, 2] Double -> NN.Embedding 4 2 T.TensorData Double
seedGradient = T.run2 T.CPU $ \indices seed -> snd (AG.pullback (`NN.embedding` indices) parameters) seed

tests :: TestTree
tests =
  testGroup
    "embedding"
    [ testProperty "lookup appends the feature axis to arbitrary-shaped indices" $ H.withTests 1 $ H.property $ do
        let indices = T.fromList @'[2, 2, 2] [3, 0, 2, 1, 1, 3, 0, 2]
            output :: T.Tensor '[2, 2, 2, 2] Double
            output = NN.embedding parameters indices
            scalar :: T.Tensor '[2] Double
            scalar = NN.embedding parameters $ T.fromList @'[] [2]
        values output H.=== [7, 8, 1, 2, 5, 6, 3, 4, 3, 4, 7, 8, 1, 2, 5, 6]
        values scalar H.=== [5, 6],
      testProperty "weighted pullback adds repeated rows and zeros unselected rows" $ H.withTests 1 $ H.property $ do
        let indices = T.fromList @'[4] [2, 0, 2, 1]
            (_, backward) = AG.pullback (`NN.embedding` indices) parameters
            gradient = backward $ T.fromList @'[4, 2] [1, 2, 3, 4, 5, 6, 7, 8]
        values (NN.embeddingWeights gradient) H.=== [3, 4, 7, 8, 6, 8, 0, 0],
      testProperty "prepared pullback agrees with a row-wise sum for changing indices and seeds" $ H.withTests 20 $ H.property $ do
        indices <- H.forAll $ Gen.list (Range.singleton 6) $ Gen.int $ Range.linear 0 3
        seeds <- H.forAll $ Gen.list (Range.singleton 12) $ Gen.int $ Range.linear (-10) 10
        let gradient = seedGradient (T.fromListData indices) (T.fromListData $ map fromIntegral seeds)
            expected = [fromIntegral $ sum [seeds !! (2 * position + column) | (position, key) <- zip [0 ..] indices, key == row] | row <- [0 .. 3], column <- [0, 1]]
        T.toList (NN.embeddingWeights gradient) H.=== expected,
      testProperty "shared tables accumulate gradients from multiple uses" $ H.withTests 1 $ H.property $ do
        let indices = T.fromList @'[3] [0, 2, 0]
            gradient = AG.gradient (\p -> T.sum $ NN.embedding p indices + NN.embedding p indices) parameters
        values (NN.embeddingWeights gradient) H.=== [4, 4, 0, 0, 2, 2, 0, 0],
      testProperty "prepared lookup accepts changing weights and integer inputs" $ H.withTests 1 $ H.property $ do
        let execute = T.run2 T.CPU $ \p selected -> NN.embedding p (selected :: T.Tensor '[2] Int)
            first = T.fromListData @'[2] [2, 0]
            second = T.fromListData @'[2] [1, 3]
            original = T.run T.CPU parameters
            doubled = T.run T.CPU $ NN.Embedding $ NN.embeddingWeights parameters * 2
        T.toList (execute original first) H.=== [5, 6, 1, 2]
        T.toList (execute doubled second) H.=== [6, 8, 14, 16],
      testProperty "prepared SGD learns selected embeddings while preserving unused rows" $ H.withTests 1 $ H.property $ do
        let initial = NN.Embedding $ T.fromList @'[4, 2] [0, 0, 10, 20, 0, 0, 30, 40 :: Double]
            targets = T.fromList @'[3, 2] [1, 2, 3, 4, 1, 2 :: Double]
            execute = T.run2 T.CPU $ \p selected ->
              let (loss, gradients) = AG.backprop (\table -> NN.mse T.mean (NN.embedding table selected) (AG.constant targets)) p
                  updated = NN.evalState (NN.optimizerStep (NN.SGD 0.5) p gradients) ()
               in (loss, updated)
            indices = T.fromListData @'[3] [0, 2, 0]
        trained <- foldM (\p _ -> pure $ snd $ execute p indices) (T.run T.CPU initial) [1 .. 12 :: Int]
        let (loss, _) = execute trained indices
            rows = T.toList $ NN.embeddingWeights trained
        H.assert $ all (< 0.1) $ T.toList loss
        take 2 (drop 2 rows) H.=== [10, 20]
        drop 6 rows H.=== [30, 40],
      testProperty "empty indices and zero-sized tables preserve valid output and gradient shapes" $ H.withTests 1 $ H.property $ do
        let empty = T.fromList @'[2, 0] []
            (output, gradient) = AG.backprop (`NN.embedding` empty) parameters
            zeroTable = NN.Embedding (T.zeros @'[0, 2] @Double)
            (zeroOutput, zeroGradient) = AG.backprop (\p -> NN.embedding p $ T.fromList @'[0] []) zeroTable
            noFeatures = NN.Embedding (T.zeros @'[3, 0] @Double)
            (noOutput, noGradient) = AG.backprop (\p -> NN.embedding p $ T.fromList @'[2] [0, 2]) noFeatures
        values output H.=== []
        values (NN.embeddingWeights gradient) H.=== replicate 8 0
        values zeroOutput H.=== []
        values (NN.embeddingWeights zeroGradient) H.=== []
        values noOutput H.=== []
        values (NN.embeddingWeights noGradient) H.=== [],
      testProperty "Float embeddings support batched repeated-index gradients" $ H.withTests 1 $ H.property $ do
        let table = NN.Embedding $ T.fromList @'[3, 2] [1, 2, 3, 4, 5, 6 :: Float]
            indices = T.fromList @'[2, 2] [2, 1, 2, 0]
            (output, gradient) = AG.backprop (`NN.embedding` indices) table
        T.toList (T.run T.CPU output) H.=== [5, 6, 3, 4, 5, 6, 1, 2]
        T.toList (T.run T.CPU $ NN.embeddingWeights gradient) H.=== [1, 1, 1, 1, 2, 2],
      testProperty "embedding composes as an arrow without changing model state" $ H.withTests 1 $ H.property $ do
        let indices = T.fromList @'[2] [1, 3]
            (output, state) = NN.forward (NN.embedding parameters) indices True
        state H.=== True
        values output H.=== [3, 4, 7, 8],
      testProperty "materialized embedding weights round-trip through named weight storage" $ H.withTests 1 $ H.property $ do
        let stored = NN.extractWeights $ T.run T.CPU parameters
            template = NN.Embedding $ T.fromListData @'[4, 2] $ replicate 8 (0 :: Double)
        map fst (Weights.weightsToList stored) H.=== [["embeddingWeights"]]
        restored <- H.evalEither $ NN.loadWeights stored template
        approxList (T.toList $ NN.embeddingWeights restored) [1, 2, 3, 4, 5, 6, 7, 8],
      testProperty "embedding rejects a mismatched output feature dimension" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/EmbeddingOutput.hs" "Couldn't match type",
      testProperty "embedding rejects non-integer index tensors" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/EmbeddingIndices.hs" "Couldn't match"
    ]
