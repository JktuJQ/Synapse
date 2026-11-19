Classifying concentric rings
===========================

Assign a point in the plane to one of three concentric rings.
Each ring has a slightly varying radius; the validation points lie between the training angles.
A single linear classifier cannot separate these classes, so we use two nonlinear hidden layers.

> {-# LANGUAGE DataKinds #-}
> {-# LANGUAGE DeriveGeneric #-}
> {-# LANGUAGE FlexibleContexts #-}
> {-# LANGUAGE FlexibleInstances #-}
> {-# LANGUAGE ImpredicativeTypes #-}
> {-# LANGUAGE ScopedTypeVariables #-}
> {-# LANGUAGE TypeApplications #-}
> {-# LANGUAGE TypeFamilies #-}
> {-# LANGUAGE UndecidableInstances #-}
> {-# LANGUAGE NoDeepSubsumption #-}
>
> module Main (main) where
>
> import Control.Arrow ((>>>))
> import Control.Exception (evaluate)
> import Control.Monad (forM_, when)
> import Control.Monad.Trans.State.Strict (get, put)
> import Data.Functor.Identity (Identity (..))
> import Data.IORef (newIORef, readIORef, writeIORef)
> import GHC.Generics (Generic)
> import GHC.TypeNats (KnownNat)
> import System.Environment (getArgs)
> import System.Random (mkStdGen)
> import Text.Printf (printf)
>
> import qualified Synapse.Autograd as AG
> import qualified Synapse.NN as NN
> import qualified Synapse.Tensor as T

The parameter record describes the layer weights and biases.
Its `TensorStructure` instance lets autograd, the optimizer, and the backend traverse the same record.

> data Parameters t = Parameters
>     { hidden :: NN.Dense 2 24 t Double
>     , middle :: NN.Dense 24 24 t Double
>     , output :: NN.Dense 24 3 t Double
>     }
>     deriving (Generic)
>
> instance T.TensorStructure (Parameters T.Tensor) where
>     type WithTensor t (Parameters T.Tensor) = Parameters t

The forward pass returns three logits per point.
Cross entropy consumes these scores directly; softmax is only needed when interpreting them as probabilities.

> predict
>     :: (KnownNat batch_size, T.RealFloatTensorOps t Double)
>     => Parameters t
>     -> t '[batch_size, 2] Double
>     -> t '[batch_size, 3] Double
> predict parameters =
>     NN.tanh . NN.dense (hidden parameters)
>         >>> NN.tanh . NN.dense (middle parameters)
>         >>> NN.dense (output parameters)

Generate equally many points from each ring.
Each sample contains two coordinates and a scalar class index: 0, 1, or 2.
The shift changes the sampling angles without changing the underlying rings.

> dataset
>     :: Int
>     -> Double
>     -> NN.Dataset Identity (T.TensorData '[2] Double, T.TensorData '[] Int)
> dataset pointsPerClass shift = NN.Indexed (3 * pointsPerClass) $ \index ->
>     let (category, position) = index `quotRem` pointsPerClass
>         angle = 2 * pi * (fromIntegral position + shift) / fromIntegral pointsPerClass
>         radius = 0.4 + 0.45 * fromIntegral category + 0.035 * sin (7 * angle + fromIntegral category)
>      in pure
>             ( T.fromListData @'[2] [radius * cos angle, radius * sin angle]
>             , T.fromListData @'[] [category]
>             )

Initialize weights with Xavier initialization and biases with zeros.
The fixed seed makes the initial model reproducible.

> initialParameters :: Parameters T.Tensor
> initialParameters = NN.evalState initialize $ mkStdGen 42
>   where
>     initialize = do
>         first <- NN.xavierUniform @'[2, 24] 2 24 1
>         second <- NN.xavierUniform @'[24, 24] 24 24 1
>         third <- NN.xavierUniform @'[24, 3] 24 3 1
>         pure $
>             Parameters
>                 (NN.Dense first $ T.zeros @'[24])
>                 (NN.Dense second $ T.zeros @'[24])
>                 (NN.Dense third $ T.zeros @'[3])

Adam retains a step counter and two moment estimates for every parameter.
Keep its state alongside the parameters between batches; only the parameters are differentiated.

> optimizer :: NN.Adam
> optimizer = NN.Adam 0.01 0.9 0.999 1e-8

The targets have shape `[batch_size]`, whereas the logits have shape `[batch_size, 3]`.
`unsafeCrossEntropy` selects each target class without checking its index.
Our dataset constructs only indices in `[0, 2]`, satisfying that requirement.
All batches contain real samples, so the loss uses an ordinary mean without a mask.

The training step updates its parameters and optimizer state through `State`, returning only the loss.
The batch executor uses `runState` to obtain the updated tensors at the execution boundary.

> type TrainingState = (Parameters T.Tensor, NN.OptimizerState NN.Adam (Parameters T.Tensor))
>
> trainingStep
>     :: KnownNat batch_size
>     => T.Tensor '[batch_size, 2] Double
>     -> T.Tensor '[batch_size] Int
>     -> NN.State TrainingState (T.Tensor '[] Double)
> trainingStep features targets = do
>     (parameters, optimizerState) <- get
>     let objective
>             :: AG.Context s
>             => Parameters (AG.Tensor s)
>             -> AG.Tensor s '[] Double
>         objective p =
>             NN.unsafeCrossEntropy @1
>                 T.mean
>                 (predict p $ AG.constant features)
>                 targets
>         (loss, gradients) = AG.backprop objective parameters
>         (updated, nextOptimizer) =
>             NN.runState (NN.optimizerStep optimizer parameters gradients) optimizerState
>     put (updated, nextOptimizer)
>     pure loss

Validation uses all held-out points at once.
Accuracy counts correct predictions; macro F1 gives each class equal weight.
The confusion matrix has actual classes in rows and predicted classes in columns.

> validationStep
>     :: KnownNat batch_size
>     => Parameters T.Tensor
>     -> T.Tensor '[batch_size, 2] Double
>     -> T.Tensor '[batch_size] Int
>     -> (T.Tensor '[] Double, T.Tensor '[] Double, T.Tensor '[] Double, T.Tensor '[3, 3] Int)
> validationStep parameters features targets =
>     let logits = predict parameters features
>      in ( NN.unsafeCrossEntropy @1 T.mean logits targets
>         , NN.accuracy @1 logits targets
>         , NN.f1 @1 NN.Macro logits targets
>         , NN.confusionMatrix @1 logits targets
>         )

Synapse tensors describe a staged computation.
To execute it, we select a backend: CPU by default, or GPU when explicitly requested.
GPU execution requires Synapse to be built with the Cabal flag -fgpu and a supported CUDA device.

> chooseBackend :: IO T.Backend
> chooseBackend = do
>     args <- getArgs
>     case args of
>         [] -> pure T.CPU
>         ["--cpu"] -> pure T.CPU
>         ["--gpu"] -> pure T.GPU
>         _ -> ioError $ userError "Usage: classification-example [--cpu | --gpu]"

Prepare the training and validation computations outside the loop.
The 288 training points divide evenly into batches of 32; validation uses one batch of 96.
Each prepared step retains a single specialization across epochs.

> main :: IO ()
> main = do
>     backend <- chooseBackend
>     let training = dataset 96 0
>         validation = dataset 32 0.5
>         trainingPlan = NN.planBatches 32 NN.KeepLast training
>         validationPlan = NN.planBatches 96 NN.KeepLast validation
>         step = NN.prepareBatchStep backend trainingPlan $ \s x y _ -> NN.runState (trainingStep x y) s
>         measure = NN.prepareBatchStep backend validationPlan $ \p x y _ -> validationStep p x y

Materialize the initial parameters and Adam state together.
Retain the updated pair and a separate shuffle generator between epochs.

>     state <- newIORef $ T.run backend (initialParameters, NN.initialState optimizer initialParameters)
>     generator <- newIORef $ mkStdGen 123
>
>     putStrLn "Three concentric rings: Dense 2 -> tanh 24 -> tanh 24 -> 3 logits; Adam."

Epoch zero evaluates the untrained model.
For subsequent epochs, shuffle sample indices and continue from the updated generator.

>     forM_ [0 .. 50 :: Int] $ \epoch -> do
>         when (epoch > 0) $ do
>             random <- readIORef generator
>             let (source, nextGenerator) = NN.runState (NN.shuffle training) random
>             writeIORef generator nextGenerator

Pass both parameters and optimizer state through the same prepared step for every batch.
Force the scalar loss so each iteration executes its computation before retaining the updated state.

>             forM_ (NN.loadBatches 32 NN.KeepLast source) $ \(Identity batch) -> do
>                 current <- readIORef state
>                 (loss, updated) <- either (fail . show) pure $ NN.runBatch step current batch
>                 _ <- evaluate $ sum $ T.toList loss
>                 writeIORef state updated

Every 20 epochs, report validation loss, accuracy, and macro F1.
At the end, print the confusion matrix.

>         when (epoch `mod` 20 == 0) $
>             forM_ (NN.batchLoads validationPlan) $ \(Identity batch) -> do
>                 (current, _) <- readIORef state
>                 (loss, accuracy, macroF1, confusion) <-
>                     either (fail . show) pure $ NN.runBatch measure current batch
>                 printf
>                     "Epoch %3d | loss %.5f | accuracy %.1f%% | macro F1 %.4f\n"
>                     epoch
>                     (sum $ T.toList loss)
>                     (100 * sum (T.toList accuracy))
>                     (sum $ T.toList macroF1)
>                 when (epoch == 100) $ do
>                     putStrLn "Confusion matrix (actual rows, predicted columns):"
>                     let counts = T.toList confusion
>                     forM_ [0 .. 2] $ \row -> print $ take 3 $ drop (3 * row) counts
