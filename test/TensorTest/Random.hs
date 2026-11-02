{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module TensorTest.Random (tests) where

import Data.Word (Word32)
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import System.Random (genWord64, mkStdGen)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data NetworkState t = NetworkState (T.Generator t) (t '[] Double)
  deriving (Generic)

type instance T.WithTensor t (NetworkState T.Tensor) = NetworkState t

tests :: TestTree
tests =
  testGroup
    "random"
    [ testProperty "Philox matches the Random123 zero-key vector" $ H.withTests 1 $ H.property $ do
        let (words_, _) = T.randomWords @'[4] $ T.generatorFromSeed 0
        T.toList (T.run T.CPU words_) H.=== ([0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8] :: [Word32]),
      testProperty "prepared sampling is reproducible and advances its input stream" $ H.withTests 1 $ H.property $ do
        let sample = T.run1 T.CPU $ T.randomWords @'[8]
            initial = T.run T.CPU $ T.generatorFromSeed 42
            (first, next) = sample initial
            (repeated, _) = sample initial
            (second, _) = sample next
            (whole, _) = T.randomWords @'[16] $ T.generatorFromSeed 42
        T.toList first H.=== T.toList repeated
        H.assert $ T.toList first /= T.toList second
        T.toList first ++ T.toList second H.=== T.toList (T.run T.CPU whole),
      testProperty "partial draws consume whole blocks and empty draws consume none" $ H.withTests 1 $ H.property $ do
        let initial = T.generatorFromSeed 7
            (short, afterShort) = T.randomWords @'[1] initial
            (block, afterBlock) = T.randomWords @'[4] initial
            (empty, afterEmpty) = T.randomWords @'[0, 3] initial
            (nextShort, _) = T.randomWords @'[4] afterShort
            (nextBlock, _) = T.randomWords @'[4] afterBlock
            (nextEmpty, _) = T.randomWords @'[4] afterEmpty
        T.toList (T.run T.CPU short) H.=== take 1 (T.toList $ T.run T.CPU block)
        T.toList (T.run T.CPU nextShort) H.=== T.toList (T.run T.CPU nextBlock)
        T.toList (T.run T.CPU empty) H.=== []
        T.toList (T.run T.CPU nextEmpty) H.=== T.toList (T.run T.CPU block),
      testProperty "host initialization consumes exactly one seed word" $ H.withTests 1 $ H.property $ do
        let host = mkStdGen 123
            (seed, nextHost) = genWord64 host
            (generator, actualHost) = T.initializeGenerator host
            (actual, _) = T.randomWords @'[8] generator
            (expected, _) = T.randomWords @'[8] $ T.generatorFromSeed seed
        actualHost H.=== nextHost
        T.toList (T.run T.CPU actual) H.=== T.toList (T.run T.CPU expected),
      testProperty "uniform samples stay in the half-open interval for both precisions" $ H.withTests 1 $ H.property $ do
        let generator = T.generatorFromSeed 42
            (floats, _) = T.uniform @'[4096] @Float generator
            (doubles, _) = T.uniform @'[4096] @Double generator
            check samples = H.assert $ all (\x -> x >= 0 && x < 1) samples
        check $ T.toList $ T.run T.CPU floats
        check $ T.toList $ T.run T.CPU doubles
        let xs = T.toList $ T.run T.CPU doubles
        H.assert $ abs (sum xs / 4096 - 0.5) < 0.025,
      testProperty "Bernoulli endpoints are exact and interior samples include both outcomes" $ H.withTests 1 $ H.property $ do
        let generator = T.generatorFromSeed 42
            (zeros, _) = T.bernoulli @'[64] 0 generator
            (ones, _) = T.bernoulli @'[64] 1 generator
            (mask, _) = T.bernoulli @'[4096] 0.25 generator
            count = length $ filter id $ T.toList $ T.run T.CPU mask
        T.toList (T.run T.CPU zeros) H.=== replicate 64 False
        T.toList (T.run T.CPU ones) H.=== replicate 64 True
        H.assert $ count > 900 && count < 1150,
      testProperty "sampling preserves row-major order across shapes" $ H.withTests 1 $ H.property $ do
        let generator = T.generatorFromSeed 5
            (matrix, _) = T.randomWords @'[2, 3] generator
            (vector, _) = T.randomWords @'[6] generator
            (scalar, _) = T.randomWords @'[] generator
        T.toList (T.run T.CPU matrix) H.=== T.toList (T.run T.CPU vector)
        T.toList (T.run T.CPU scalar) H.=== take 1 (T.toList $ T.run T.CPU vector),
      testProperty "generator records pass through runN and backprop auxiliary results" $ H.withTests 1 $ H.property $ do
        let ((loss, state), gradient) =
              AG.backprop
                ( \x ->
                    let (mask, next) = T.bernoulli @'[8] 0.5 $ T.generatorFromSeed 42
                     in (T.sum $ T.select mask (x * 2) 0, AG.constant $ NetworkState next 17)
                )
                (1 :: T.Tensor '[8] Double)
            (computedLoss, (NetworkState generator marker, computedGradient)) = T.run T.CPU (loss, (state, gradient))
            (sample, _) = T.run1 T.CPU (T.randomWords @'[4]) generator
            (_, advanced) = T.randomWords @'[8] $ T.generatorFromSeed 42
            (expected, _) = T.randomWords @'[4] advanced
        T.toList computedLoss H.=== [sum $ T.toList computedGradient]
        T.toList marker H.=== [17]
        T.toList sample H.=== T.toList (T.run T.CPU expected)
    ]
