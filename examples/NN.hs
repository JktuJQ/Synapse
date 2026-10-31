{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module Main (main) where

import Control.Arrow ((>>>))
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
  { hidden :: NN.Dense 1 32 t Double,
    middle :: NN.Dense 32 32 t Double,
    output :: NN.Dense 32 1 t Double
  }
  deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor) = Parameters t

predict ::
  ( KnownNat batch_size,
    T.TensorOps t Double,
    Floating (t '[batch_size, 32] Double),
    Num (t '[batch_size, 1] Double),
    T.Ord (t '[batch_size, 32] Double),
    T.Conditional (t '[batch_size, 32] Double),
    Condition (t '[batch_size, 32] Double) ~ Comparison (t '[batch_size, 32] Double)
  ) =>
  Parameters t ->
  t '[batch_size, 1] Double ->
  t '[batch_size, 1] Double
predict parameters =
  NN.relu . NN.dense (hidden parameters)
    >>> NN.tanh . NN.dense (middle parameters)
    >>> NN.dense (output parameters)

targetFunction :: Double -> Double
targetFunction x = sin (2 * pi * x) + 0.3 * cos (5 * pi * x) + 0.2 * x * x + 0.5 * max 0 (x - 0.2)

dataset :: Int -> Double -> NN.Dataset Identity (T.TensorData '[1] Double, T.TensorData '[1] Double)
dataset count shift = NN.Indexed count $ \index ->
  let x = -1 + 2 * (fromIntegral index + shift) / fromIntegral count
   in pure (T.fromListData @'[1] [x], T.fromListData @'[1] [targetFunction x])

initialParameters :: Parameters T.Tensor
initialParameters =
  let (first, rest) = splitGen (mkStdGen 7)
      (second, third) = splitGen rest
   in Parameters
        (NN.Dense (NN.xavierUniform @'[1, 32] 1 32 (sqrt 2) first) (T.zeros @'[32]))
        (NN.Dense (NN.xavierUniform @'[32, 32] 32 32 1 second) (T.zeros @'[32]))
        (NN.Dense (NN.xavierUniform @'[32, 1] 32 1 1 third) (T.zeros @'[1]))

trainingStep ::
  forall batch_size.
  (KnownNat batch_size) =>
  Parameters T.Tensor ->
  T.Tensor '[batch_size, 1] Double ->
  T.Tensor '[batch_size, 1] Double ->
  T.Tensor '[batch_size] Bool ->
  (T.Tensor '[] Double, Parameters T.Tensor)
trainingStep parameters features targets mask =
  let weights = T.select (T.unsqueeze @1 mask) 1 0 :: T.Tensor '[batch_size, 1] Double
      objective :: (AG.Context s) => Parameters (AG.Tensor s) -> AG.Tensor s '[] Double
      objective p = NN.mse (\errors -> T.sum (errors * AG.constant weights) / AG.constant (T.sum weights)) (predict p (AG.constant features)) (AG.constant targets)
      (loss, gradients) = AG.backprop objective parameters
      (updated, ()) = NN.optimizerStep (NN.SGD 0.02) parameters gradients ()
   in (loss, updated)

validationStep ::
  (KnownNat batch_size) =>
  Parameters T.Tensor ->
  T.Tensor '[batch_size, 1] Double ->
  T.Tensor '[batch_size, 1] Double ->
  (T.Tensor '[] Double, (T.Tensor '[] Double, T.Tensor '[] Double))
validationStep parameters features targets =
  let predictions = predict parameters features
   in (NN.mse T.mean predictions targets, (NN.mae T.mean predictions targets, NN.r2 predictions targets))

main :: IO ()
main = do
  let training = dataset 273 0
      validation = dataset 128 0.5

  parameters <- newIORef (T.run T.CPU initialParameters)
  generator <- newIORef (mkStdGen 42)

  putStrLn "Dense 1 -> ReLU 32 -> tanh 32 -> Dense 1"

  forM_ [0 .. 1000 :: Int] $ \epoch -> do
    when (epoch > 0) $ do
      random <- readIORef generator
      let (source, nextGenerator) = NN.shuffle training random
      writeIORef generator nextGenerator

      forM_ (NN.loadBatches 32 NN.KeepLast source) $ \load ->
        NN.withBatch (runIdentity load) $ \batch -> do
          current <- readIORef parameters
          let (loss, updated) = T.run4 T.CPU trainingStep current (NN.batchInputs batch) (NN.batchTargets batch) (NN.batchMask batch)
          _ <- evaluate (sum (T.toList loss))
          writeIORef parameters updated

    when (epoch `mod` 50 == 0) $
      forM_ (NN.loadBatches 128 NN.KeepLast validation) $ \load ->
        NN.withBatch (runIdentity load) $ \batch -> do
          current <- readIORef parameters
          let (mse, (mae, r2)) = T.run3 T.CPU validationStep current (NN.batchInputs batch) (NN.batchTargets batch)
          printf
            "Epoch %4d | MSE %.5f | MAE %.5f | R² %.5f\n"
            epoch
            (sum (T.toList mse))
            (sum (T.toList mae))
            (sum (T.toList r2))
