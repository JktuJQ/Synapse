{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Main (main) where

import Control.Arrow (Arrow, arr, (>>>))
import Control.Exception (evaluate)
import Control.Monad (foldM, forM_, when)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat)
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import System.IO (BufferMode (LineBuffering), hSetBuffering, stdout)
import System.Random (mkStdGen, splitGen)
import Text.Printf (printf)

data Parameters t = Parameters
  { tokens :: NN.Embedding 8 8 t Double,
    selfAttention :: NN.MultiHeadAttention 8 2 t Double,
    attentionNorm :: NN.LayerNorm '[8] t Double,
    feedForwardNorm :: NN.LayerNorm '[8] t Double,
    expand :: NN.Dense 8 16 t Double,
    project :: NN.Dense 16 8 t Double
  }
  deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor) = Parameters t

-- Rotate projected Q/K independently in each head; V is unchanged.
rotaryAttention ::
  forall batch_size t arrow.
  ( KnownNat batch_size,
    Arrow arrow,
    T.TensorOps t Double,
    Num (t '[batch_size, 6, 8] Double),
    Num (t '[batch_size, 2, 6, 4] Double),
    Fractional (t '[batch_size, 2, 6, 6] Double)
  ) =>
  NN.MultiHeadAttention 8 2 t Double ->
  arrow (t '[batch_size, 6, 8] Double) (t '[batch_size, 6, 8] Double)
rotaryAttention parameters = proc input -> do
  projectedQ <- NN.dense (NN.attentionQuery parameters) -< input
  projectedK <- NN.dense (NN.attentionKey parameters) -< input
  projectedV <- NN.dense (NN.attentionValue parameters) -< input
  let splitHeads = T.permuteAxes @'[0, 2, 1, 3] . T.reshape @'[batch_size, 6, 2, 4]
      query = NN.rope 10000 0 $ splitHeads projectedQ
      key = NN.rope 10000 0 $ splitHeads projectedK
      value = splitHeads projectedV
      mask = T.broadcast $ NN.causalMask @6 @6
      attended = NN.attention (Just mask) query key value
      joined = T.reshape @'[batch_size, 6, 8] $ T.permuteAxes @'[0, 2, 1, 3] attended
  NN.dense (NN.attentionOutput parameters) -< joined

predict ::
  ( KnownNat batch_size,
    Arrow arrow,
    T.TensorOps t Double,
    Floating (t '[batch_size, 6, 8] Double),
    Floating (t '[batch_size, 6, 16] Double),
    Num (t '[batch_size, 2, 6, 4] Double),
    Fractional (t '[batch_size, 2, 6, 6] Double)
  ) =>
  Parameters t ->
  arrow (T.Tensor '[batch_size, 6] Int) (t '[batch_size, 6, 8] Double)
predict parameters = proc input -> do
  embedded <- NN.embedding (tokens parameters) -< input
  normalized <- NN.layerNorm 1e-5 (attentionNorm parameters) -< embedded
  attended <- rotaryAttention (selfAttention parameters) -< normalized
  let residual = embedded + attended
  normalizedAgain <- NN.layerNorm 1e-5 (feedForwardNorm parameters) -< residual
  expanded <- NN.dense (expand parameters) >>> NN.tanh -< normalizedAgain
  projected <- NN.dense (project parameters) -< expanded
  -- The embedding table is also the output projection: one shared parameter.
  arr (\x -> T.matmul x $ T.transpose $ NN.embeddingWeights $ tokens parameters) -< residual + projected

-- Repeat a three-token motif. Predicting the next token needs earlier context:
-- e.g. 2,5,1,2,5,1 -> 5,1,2,5,1,2.
-- Multiplication permutes the 512 possible motifs without repetitions.
sample :: Int -> ([Int], [Int])
sample index =
  let code = (73 * index) `mod` 512
      motif = [code `div` 64, (code `div` 8) `mod` 8, code `mod` 8]
      sequenceTokens = cycle motif
   in (take 6 sequenceTokens, take 6 $ drop 1 sequenceTokens)

initialParameters :: Parameters T.Tensor
initialParameters =
  let (embeddingGen, rest) = splitGen $ mkStdGen 7
      (queryGen, rest1) = splitGen rest
      (keyGen, rest2) = splitGen rest1
      (valueGen, rest3) = splitGen rest2
      (outputGen, rest4) = splitGen rest3
      (expandGen, projectGen) = splitGen rest4
   in Parameters
        (NN.Embedding $ NN.normal 0 0.2 embeddingGen)
        ( NN.MultiHeadAttention
            (NN.Dense (NN.xavierUniform 8 8 1 queryGen) 0)
            (NN.Dense (NN.xavierUniform 8 8 1 keyGen) 0)
            (NN.Dense (NN.xavierUniform 8 8 1 valueGen) 0)
            (NN.Dense (NN.xavierUniform 8 8 1 outputGen) 0)
        )
        (NN.LayerNorm 1 0)
        (NN.LayerNorm 1 0)
        (NN.Dense (NN.xavierUniform 8 16 1 expandGen) 0)
        (NN.Dense (NN.xavierUniform 16 8 1 projectGen) 0)

optimizer :: NN.AdamW
optimizer = NN.AdamW (NN.Adam 0.01 0.9 0.999 1e-8) 0.01

-- The first two next-token targets are not determined by the observed prefix.
-- Train and measure only positions where the whole motif is already visible.
trainingStep ::
  (KnownNat batch_size) =>
  Parameters T.Tensor ->
  NN.OptimizerState NN.AdamW (Parameters T.Tensor) ->
  T.Tensor '[batch_size, 6] Int ->
  T.Tensor '[batch_size, 6] Int ->
  (T.Tensor '[] Double, (Parameters T.Tensor, NN.OptimizerState NN.AdamW (Parameters T.Tensor)))
trainingStep parameters optimizerState features targets =
  let (loss, gradients) =
        AG.backprop
          ( \p ->
              NN.unsafeCrossEntropy @2
                T.mean
                (T.slice @'[T.All, T.Range 2 6 1, T.All] $ predict p features)
                (T.slice @'[T.All, T.Range 2 6 1] targets)
          )
          parameters
      updated = NN.runState (NN.optimizerStep optimizer parameters gradients) optimizerState
   in (loss, updated)

measure ::
  (KnownNat batch_size) =>
  Parameters T.Tensor ->
  T.Tensor '[batch_size, 6] Int ->
  T.Tensor '[batch_size, 6] Int ->
  (T.Tensor '[] Double, T.Tensor '[] Double)
measure parameters features targets =
  let logits = T.slice @'[T.All, T.Range 2 6 1, T.All] $ predict parameters features
      labels = T.slice @'[T.All, T.Range 2 6 1] targets
   in (NN.unsafeCrossEntropy @2 T.mean logits labels, NN.accuracy @2 logits labels)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  let (trainingInputs, trainingLabels) = unzip $ map sample [0 .. 127]
      (validationInputs, validationLabels) = unzip $ map sample [400 .. 431]
      features = T.fromListData @'[128, 6] $ concat trainingInputs
      targets = T.fromListData @'[128, 6] $ concat trainingLabels
      validationFeatures = T.fromListData @'[32, 6] $ concat validationInputs
      validationTargets = T.fromListData @'[32, 6] $ concat validationLabels
      step = T.run4 T.CPU $ trainingStep @128
      check = T.run3 T.CPU $ measure @32
      initial = T.run T.CPU (initialParameters, NN.initialState optimizer initialParameters)
      report iteration parameters = do
        let (loss, accuracy) = check parameters validationFeatures validationTargets
        printf "Step %3d | validation loss %.4f | accuracy %.1f%%\n" (iteration :: Int) (sum $ T.toList loss) (100 * sum (T.toList accuracy))

  putStrLn "Next-token prediction: repeat a three-token motif from an eight-token vocabulary."
  putStrLn "Embedding -> causal attention with RoPE -> MLP; pre-norm, residuals, tied output weights, AdamW."

  -- This tiny dataset fits in one batch. Prepare the step once and keep passing
  -- the current parameters and optimizer state as runtime inputs.
  report 0 $ fst initial
  (trained, _) <-
    foldM
      ( \(parameters, optimizerState) iteration -> do
          let (loss, updated) = step parameters optimizerState features targets
          _ <- evaluate $ sum $ T.toList loss
          when (iteration `mod` 25 == 0) $ report iteration $ fst updated
          pure updated
      )
      initial
      [1 .. 200 :: Int]

  putStrLn "\nHeld-out motifs (next-token predictions after observing the first three tokens):"
  forM_ [400, 407, 419, 431] $ \index -> do
    let (input, labels) = sample index
        predicted = T.run2 T.CPU (\p x -> T.argmax @2 $ predict p x) trained $ T.fromListData @'[1, 6] input
    putStrLn $ "Input:     " ++ show input
    putStrLn $ "Expected:  " ++ show (drop 2 labels)
    putStrLn $ "Predicted: " ++ show (drop 2 $ T.toList predicted)
