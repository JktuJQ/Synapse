{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Main (main) where

import Control.Arrow (arr)
import Control.Exception (evaluate)
import Control.Monad (forM_, when)
import Data.Functor.Identity (Identity, runIdentity)
import Data.IORef (newIORef, readIORef, writeIORef)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat)
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import Synapse.Tensor.Ops.Elementwise (Comparison, Condition)
import System.Random (mkStdGen, splitGen)
import Text.Printf (printf)

data Parameters t = Parameters
  { convolution :: NN.Convolution 1 2 '[3, 3] 1 t Double,
    batchNorm :: NN.BatchNorm 2 t Double,
    hidden :: NN.Dense 32 8 t Double,
    layerNorm :: NN.LayerNorm '[8] t Double,
    output :: NN.Dense 8 2 t Double
  }
  deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor) = Parameters t

data ModelState t = ModelState
  { statistics :: NN.BatchNormState 2 t Double,
    randomness :: NN.DropoutState t
  }
  deriving (Generic)

type instance T.WithTensor t (ModelState T.Tensor) = ModelState t

statisticsLens :: (Functor f) => (NN.BatchNormState 2 t Double -> f (NN.BatchNormState 2 t Double)) -> ModelState t -> f (ModelState t)
statisticsLens change state = fmap (\updated -> state {statistics = updated}) $ change $ statistics state

randomnessLens :: (Functor f) => (NN.DropoutState t -> f (NN.DropoutState t)) -> ModelState t -> f (ModelState t)
randomnessLens change state = fmap (\updated -> state {randomness = updated}) $ change $ randomness state

predict ::
  forall batch_size t.
  ( KnownNat batch_size,
    T.TensorOps t Double,
    Floating (t '[batch_size, 2, 8, 8] Double),
    T.Ord (t '[batch_size, 2, 8, 8] Double),
    T.Conditional (t '[batch_size, 2, 8, 8] Double),
    Condition (t '[batch_size, 2, 8, 8] Double) ~ Comparison (t '[batch_size, 2, 8, 8] Double),
    Floating (t '[batch_size, 8] Double),
    T.Conditional (t '[batch_size, 8] Double),
    Condition (t '[batch_size, 8] Double) ~ T.Tensor '[batch_size, 8] Bool,
    Num (t '[batch_size, 2] Double)
  ) =>
  NN.ForwardMode ->
  Parameters t ->
  NN.Forward (ModelState T.Tensor) (t '[batch_size, 1, 8, 8] Double) (t '[batch_size, 2] Double)
predict mode parameters = proc input -> do
  convolved <- NN.conv2d @'[1, 1] @'[1, 1, 1, 1] @'[1, 1] (convolution parameters) -< input
  normalized <- NN.stateful statisticsLens (NN.batchNorm @1 mode 1e-5 0.1 $ batchNorm parameters) -< convolved
  activated <- NN.relu -< normalized
  flattened <- arr (T.reshape @'[batch_size, 32] . T.maxPool2d @'[2, 2] @'[2, 2] @'[0, 0, 0, 0] @'[1, 1]) -< activated
  projected <- NN.dense (hidden parameters) -< flattened
  rescaled <- NN.layerNorm 1e-5 (layerNorm parameters) -< projected
  nonlinear <- NN.tanh -< rescaled
  dropped <- NN.stateful randomnessLens (NN.dropout mode 0.1) -< nonlinear
  NN.dense (output parameters) -< dropped

dataset :: Int -> Int -> NN.Dataset Identity (T.TensorData '[1, 8, 8] Double, T.TensorData '[] Int)
dataset count offset = NN.Indexed count $ \index ->
  let sample = index + offset
      label = sample `mod` 2
      position = 1 + (sample `div` 2) `mod` 6
      pixel row column =
        let coordinate = if label == 0 then row else column
            bar = if coordinate == position || coordinate == position + 1 then 1 else 0
            noise = 0.08 * sin (fromIntegral $ sample * 17 + row * 11 + column * 7)
         in bar + noise
   in pure (T.fromListData @'[1, 8, 8] [pixel row column | row <- [0 .. 7], column <- [0 .. 7]], T.fromListData @'[] [label])

initialParameters :: Parameters T.Tensor
initialParameters =
  let (first, rest) = splitGen $ mkStdGen 7
      (second, third) = splitGen rest
   in Parameters
        (NN.Convolution (NN.kaimingNormal @'[2, 1, 3, 3] 9 0 first) 0)
        (NN.BatchNorm 1 0)
        (NN.Dense (NN.xavierUniform @'[32, 8] 32 8 1 second) 0)
        (NN.LayerNorm 1 0)
        (NN.Dense (NN.xavierUniform @'[8, 2] 8 2 1 third) 0)

initialModelState :: ModelState T.Tensor
initialModelState = ModelState NN.initialBatchNormState (NN.DropoutState $ T.generatorFromSeed 43)

optimizer :: NN.RMSprop
optimizer = NN.RMSprop 0.003 0.9 1e-8

trainingStep ::
  (KnownNat batch_size) =>
  Parameters T.Tensor ->
  ModelState T.Tensor ->
  NN.OptimizerState NN.RMSprop (Parameters T.Tensor) ->
  T.Tensor '[batch_size, 1, 8, 8] Double ->
  T.Tensor '[batch_size] Int ->
  (T.Tensor '[] Double, (Parameters T.Tensor, (ModelState T.Tensor, NN.OptimizerState NN.RMSprop (Parameters T.Tensor))))
trainingStep parameters state optimizerState features targets =
  let ((loss, nextModel), gradients) =
        AG.backprop
          ( \p ->
              let (logits, next) = NN.runState (NN.forward (predict NN.Train p) $ AG.constant features) state
               in (NN.unsafeCrossEntropy @1 T.mean logits targets, AG.constant next)
          )
          parameters
      (updated, nextOptimizer) = NN.runState (NN.optimizerStep optimizer parameters gradients) optimizerState
   in (loss, (updated, (nextModel, nextOptimizer)))

validationStep ::
  (KnownNat batch_size) =>
  Parameters T.Tensor ->
  ModelState T.Tensor ->
  T.Tensor '[batch_size, 1, 8, 8] Double ->
  T.Tensor '[batch_size] Int ->
  (T.Tensor '[] Double, T.Tensor '[] Double)
validationStep parameters state features targets =
  let logits = NN.evalState (NN.forward (predict NN.Eval parameters) features) state
   in (NN.unsafeCrossEntropy @1 T.mean logits targets, NN.accuracy @1 logits targets)

main :: IO ()
main = do
  let training = dataset 128 0
      validation = dataset 32 1000
      trainingPlan = NN.planBatches 16 NN.DropLast training
      validationPlan = NN.planBatches 32 NN.KeepLast validation
      step = NN.prepareBatchStep T.CPU trainingPlan $ \(p, (s, o)) x y _ -> trainingStep p s o x y
      measure = NN.prepareBatchStep T.CPU validationPlan $ \(p, s) x y _ -> validationStep p s x y

  model <- newIORef $ T.run T.CPU (initialParameters, (initialModelState, NN.initialState optimizer initialParameters))
  generator <- newIORef $ mkStdGen 42

  putStrLn "Horizontal vs vertical bars: Conv -> BatchNorm -> ReLU -> Pool -> Dense -> LayerNorm -> tanh -> Dropout -> Dense"
  putStrLn "RMSprop; training updates normalization statistics and dropout state, evaluation preserves them."

  forM_ [0 .. 60 :: Int] $ \epoch -> do
    when (epoch > 0) $ do
      random <- readIORef generator
      let (source, nextGenerator) = NN.shuffle training random
      writeIORef generator nextGenerator

      forM_ (NN.batchLoads $ NN.planBatches 16 NN.DropLast source) $ \load -> do
        current <- readIORef model
        (loss, updated) <- either (fail . show) pure $ NN.runBatch step current $ runIdentity load
        _ <- evaluate $ sum $ T.toList loss
        writeIORef model updated

    when (epoch `mod` 10 == 0) $
      forM_ (NN.batchLoads validationPlan) $ \load -> do
        (parameters, (state, _)) <- readIORef model
        (loss, accuracy) <- either (fail . show) pure $ NN.runBatch measure (parameters, state) $ runIdentity load
        printf "Epoch %2d | loss %.4f | accuracy %.1f%%\n" epoch (sum $ T.toList loss) (100 * sum (T.toList accuracy))
