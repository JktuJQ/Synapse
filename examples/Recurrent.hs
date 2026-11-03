{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

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
import System.Random (mkStdGen, randomRs, splitGen)
import Text.Printf (printf)

data Parameters t = Parameters
  { memory :: NN.LSTM 1 8 t Double,
    output :: NN.Dense 8 1 t Double
  }
  deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor) = Parameters t

-- The loader is batch-first; recurrent layers are time-first.
predict ::
  forall batch_size t.
  (KnownNat batch_size, T.TensorOps t Double, Num (t '[8, batch_size, 1] Double)) =>
  Parameters t ->
  NN.Forward (NN.LSTMState batch_size 8 t Double) (t '[batch_size, 8, 1] Double) (t '[batch_size, 8, 1] Double)
predict parameters = proc input -> do
  timeMajor <- arr (T.permuteAxes @'[1, 0, 2]) -< input
  hidden <- NN.Forward (NN.lstm $ memory parameters) -< timeMajor
  projected <- NN.dense (output parameters) -< hidden
  arr (T.permuteAxes @'[1, 0, 2]) -< projected

-- Predict a moving average: the answer depends on previous inputs, not just x_t.
-- Each sample is an independent sequence, so its initial recurrent state is zero.
sample :: Int -> ([Double], [Double])
sample index =
  let inputs = take 8 $ randomRs (-0.8, 0.8) $ mkStdGen index
      targets = drop 1 $ scanl (\previous input -> 0.7 * previous + 0.3 * input) 0 inputs
   in (inputs, targets)

dataset :: Int -> Int -> NN.Dataset Identity (T.TensorData '[8, 1] Double, T.TensorData '[8, 1] Double)
dataset count offset = NN.Indexed count $ \index ->
  let (inputs, targets) = sample $ index + offset
   in pure (T.fromListData inputs, T.fromListData targets)

initialParameters :: Parameters T.Tensor
initialParameters =
  let (first, rest) = splitGen $ mkStdGen 7
      (second, third) = splitGen rest
   in Parameters
        (NN.LSTM (NN.xavierUniform 1 32 1 first) (NN.xavierUniform 8 32 1 second) 0 0)
        (NN.Dense (NN.xavierUniform 8 1 1 third) 0)

optimizer :: NN.Adam
optimizer = NN.Adam 0.01 0.9 0.999 1e-8

trainingStep ::
  forall batch_size.
  (KnownNat batch_size) =>
  Parameters T.Tensor ->
  NN.OptimizerState NN.Adam (Parameters T.Tensor) ->
  T.Tensor '[batch_size, 8, 1] Double ->
  T.Tensor '[batch_size, 8, 1] Double ->
  (T.Tensor '[] Double, (Parameters T.Tensor, NN.OptimizerState NN.Adam (Parameters T.Tensor)))
trainingStep parameters optimizerState features targets =
  let (loss, gradients) =
        AG.backprop
          ( \p ->
              let predictions = NN.evalState (NN.forward (predict p) $ AG.constant features) $ NN.LSTMState 0 0
               in NN.mse T.mean predictions $ AG.constant targets
          )
          parameters
      clipped = NN.clipGradNormL2 1 gradients
      (updated, nextOptimizer) = NN.runState (NN.optimizerStep optimizer parameters clipped) optimizerState
   in (loss, (updated, nextOptimizer))

main :: IO ()
main = do
  let training = dataset 128 0
      validation = dataset 32 1000
      trainingPlan = NN.planBatches 16 NN.DropLast training
      validationPlan = NN.planBatches 32 NN.DropLast validation
      step = NN.prepareBatchStep T.CPU trainingPlan $ \(p, s) x y _ -> trainingStep p s x y
      measure = NN.prepareBatchStep T.CPU validationPlan $ \p x y _ ->
        NN.mse T.mean (NN.evalState (NN.forward (predict p) x) $ NN.LSTMState 0 0) y

  model <- newIORef $ T.run T.CPU (initialParameters, NN.initialState optimizer initialParameters)
  generator <- newIORef $ mkStdGen 42

  putStrLn "Moving average: LSTM (8 hidden units) -> Dense; Adam."
  putStrLn "Independent sequences of 8 steps; recurrent state resets for each batch."

  forM_ [0 .. 100 :: Int] $ \epoch -> do
    when (epoch > 0) $ do
      random <- readIORef generator
      let (source, nextGenerator) = NN.shuffle training random
      writeIORef generator nextGenerator
      forM_ (NN.batchLoads $ NN.planBatches 16 NN.DropLast source) $ \load -> do
        current <- readIORef model
        (loss, updated) <- either (fail . show) pure $ NN.runBatch step current $ runIdentity load
        _ <- evaluate $ sum $ T.toList loss
        writeIORef model updated

    when (epoch `mod` 20 == 0) $
      forM_ (NN.batchLoads validationPlan) $ \load -> do
        (parameters, _) <- readIORef model
        loss <- either (fail . show) pure $ NN.runBatch measure parameters $ runIdentity load
        printf "Epoch %3d | validation MSE %.6f\n" epoch (sum $ T.toList loss)

  -- The same trained weights also work one step at a time. State is carried
  -- between calls; resetting it here would lose the moving-average history.
  (parameters, _) <- readIORef model
  let (inputs, targets) = sample 2000
      cell = T.run3 T.CPU $ \p s x ->
        let (hidden, next) = NN.runState (NN.lstmCell (memory p) x) s
         in (NN.dense (output p) hidden, next)
  recurrentState <- newIORef $ T.run T.CPU (NN.LSTMState 0 0 :: NN.LSTMState 1 8 T.Tensor Double)
  putStrLn "\nStreaming inference on a new sequence:"
  putStrLn " input   target  predicted"
  forM_ (zip inputs targets) $ \(input, target) -> do
    state <- readIORef recurrentState
    let (prediction, nextState) = cell parameters state $ T.fromListData @'[1, 1] [input]
    _ <- evaluate $ sum $ T.toList prediction
    writeIORef recurrentState nextState
    printf "% .3f  % .3f  % .3f\n" input target (sum $ T.toList prediction)
