Approximating a nonlinear function
=====================================

Train a multilayer neural network to approximate a function on [-1, 1].
The target combines periodic, quadratic, and piecewise-linear behavior.
A separate set of points measures how well the model generalizes.

> {-# LANGUAGE ImpredicativeTypes #-}
> {-# LANGUAGE TypeFamilies #-}
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
> import qualified Synapse.Autograd as AG
> import qualified Synapse.NN as NN
> import qualified Synapse.Tensor as T
> import Synapse.Tensor.Ops.Logic (Comparison, Condition)
> import System.Random (mkStdGen)
> import System.Environment (getArgs)
> import Text.Printf (printf)

The model has three dense layers.
Parameters describe the tensors in each layer;
`TensorStructure` lets autograd and the backend traverse the whole record.

> data Parameters t = Parameters
>     { hidden :: NN.Dense 1 32 t Double
>     , middle :: NN.Dense 32 32 t Double
>     , output :: NN.Dense 32 1 t Double
>     }
>     deriving (Generic)
>
> instance T.TensorStructure (Parameters T.Tensor) where
>     type WithTensor t (Parameters T.Tensor) = Parameters t

The forward pass composes dense layers with ReLU and tanh activations.
The batch size is part of the input and output tensor shapes.

> predict
>     :: ( KnownNat batch_size
>        , T.RealFloatTensorOps t Double
>        , Condition (t '[batch_size, 32] Double) ~ Comparison (t '[batch_size, 32] Double)
>        )
>     => Parameters t
>     -> t '[batch_size, 1] Double
>     -> t '[batch_size, 1] Double
> predict parameters =
>     NN.relu . NN.dense (hidden parameters)
>         >>> NN.tanh . NN.dense (middle parameters)
>         >>> NN.dense (output parameters)

Generate training and validation examples from the target function.
The validation points are shifted relative to the training points.

> targetFunction
>     :: Double
>     -> Double
> targetFunction x = sin (2 * pi * x) + 0.3 * cos (5 * pi * x) + 0.2 * x * x + 0.5 * max 0 (x - 0.2)
>
> dataset
>     :: Int
>     -> Double
>     -> NN.Dataset Identity (T.TensorData '[1] Double, T.TensorData '[1] Double)
> dataset count shift = NN.Indexed count $ \index ->
>     let x = -1 + 2 * (fromIntegral index + shift) / fromIntegral count
>      in pure (T.fromListData @'[1] [x], T.fromListData @'[1] [targetFunction x])

Initialize weights with Xavier initialization and biases with zeros.
A fixed seed makes the example reproducible.

> initialParameters :: Parameters T.Tensor
> initialParameters = NN.evalState initialize $ mkStdGen 42
>   where
>     initialize = do
>         first <- NN.xavierUniform @'[1, 32] 1 32 (sqrt 2)
>         second <- NN.xavierUniform @'[32, 32] 32 32 1
>         third <- NN.xavierUniform @'[32, 1] 32 1 1
>         pure $
>             Parameters
>                 (NN.Dense first $ T.zeros @'[32])
>                 (NN.Dense second $ T.zeros @'[32])
>                 (NN.Dense third $ T.zeros @'[1])

A training step computes a masked mean squared error and differentiates it with respect to the parameters.
Expand the batch mask to match the predictions; `maskedMean` averages only the selected errors.
SGD uses the resulting gradients to update them.

The training step updates its parameters through `State`, returning only the loss.
The batch executor uses `runState` to obtain the updated parameters at the execution boundary.

> trainingStep
>     :: KnownNat batch_size
>     => T.Tensor '[batch_size, 1] Double
>     -> T.Tensor '[batch_size, 1] Double
>     -> T.Tensor '[batch_size] Bool
>     -> NN.State (Parameters T.Tensor) (T.Tensor '[] Double)
> trainingStep features targets mask = do
>     parameters <- get
>     let objective
>             :: AG.Context s
>             => Parameters (AG.Tensor s)
>             -> AG.Tensor s '[] Double
>         objective p =
>             NN.mse
>                 (T.maskedMean $ T.unsqueeze @1 mask)
>                 (predict p (AG.constant features))
>                 (AG.constant targets)
>         (loss, gradients) = AG.backprop objective parameters
>         updated = NN.evalState (NN.optimizerStep (NN.SGD 0.02) parameters gradients) ()
>     put updated
>     pure loss

Validation reports mean squared error, mean absolute error, and R^2.

> validationStep
>     :: KnownNat batch_size
>     => Parameters T.Tensor
>     -> T.Tensor '[batch_size, 1] Double
>     -> T.Tensor '[batch_size, 1] Double
>     -> (T.Tensor '[] Double, T.Tensor '[] Double, T.Tensor '[] Double)
> validationStep parameters features targets =
>     let predictions = predict parameters features
>      in (NN.mse T.mean predictions targets, NN.mae T.mean predictions targets, NN.r2 predictions targets)

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
>         _ -> ioError $ userError "Usage: regression-example [--cpu | --gpu]"

Prepare the training and validation steps outside the loop.
`KeepLast` may produce a smaller final batch, so `prepareBatchStep` retains a computation per batch size;
although internal cache can reuse compiled kernels across different shapes.
Due to that, it may be preferable to select other batching strategies,
or make sure that `KeepLast` does not affect anything.

Each epoch shuffles the dataset, runs the prepared step for each batch, and
periodically evaluates the current parameters on the validation data.

> main :: IO ()
> main = do
>     backend <- chooseBackend

Keep the prepared training and validation computations available throughout the loop.
The training set leaves a final batch of 17 samples, exercising both batch sizes.

>     let training = dataset 273 0
>         validation = dataset 128 0.5
>         trainingPlan = NN.planBatches 32 NN.KeepLast training
>         validationPlan = NN.planBatches 128 NN.KeepLast validation
>         step = NN.prepareBatchStep backend trainingPlan $ \p x y mask -> NN.runState (trainingStep x y mask) p
>         measure = NN.prepareBatchStep backend validationPlan $ \p x y _ -> validationStep p x y

Materialize the initial parameters on the selected backend.
The references retain the latest parameters and shuffle generator between epochs.

>     parameters <- newIORef (T.run backend initialParameters)
>     generator <- newIORef (mkStdGen 42)
>
>     putStrLn "Dense 1 -> ReLU 32 -> tanh 32 -> Dense 1"

Epoch zero measures the untrained model.
Subsequent epochs shuffle the training samples and retain the updated generator for the next shuffle.

>     forM_ [0 .. 50 :: Int] $ \epoch -> do
>         when (epoch > 0) $ do
>             random <- readIORef generator
>             let (source, nextGenerator) = NN.runState (NN.shuffle training) random
>             writeIORef generator nextGenerator

Load each batch and pass the current parameters to the prepared training step.
Shuffling preserves the batch sizes, so the same prepared step handles each epoch's loading actions.
Force the scalar loss before retaining the updated parameters, so each iteration executes its computation.

>             forM_ (NN.loadBatches 32 NN.KeepLast source) $ \(Identity batch) -> do
>                 current <- readIORef parameters
>                 (loss, updated) <- either (fail . show) pure $ NN.runBatch step current batch
>                 _ <- evaluate (sum (T.toList loss))
>                 writeIORef parameters updated

Every 50 epochs, evaluate the current model on the same validation points without updating its parameters.
Print all three metrics to track both prediction error and explained variance.

>         when (epoch `mod` 50 == 0) $
>             forM_ (NN.batchLoads validationPlan) $ \(Identity batch) -> do
>                 current <- readIORef parameters
>                 (mse, mae, r2) <- either (fail . show) pure $ NN.runBatch measure current batch
>                 printf
>                     "Epoch %4d | MSE %.5f | MAE %.5f | R^2 %.5f\n"
>                     epoch
>                     (sum (T.toList mse))
>                     (sum (T.toList mae))
>                     (sum (T.toList r2))
