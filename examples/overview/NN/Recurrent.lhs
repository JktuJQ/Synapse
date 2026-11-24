Learning a moving average with an LSTM
====================================

Predict a smoothed signal from a sequence of eight inputs.
The target follows y_t = 0.7 * y_(t-1) + 0.3 * x_t, starting from y_(-1) = 0.
A prediction depends on earlier inputs, so the model must retain information across time steps.

> {-# LANGUAGE Arrows #-}
> {-# LANGUAGE ImpredicativeTypes #-}
> {-# LANGUAGE TypeFamilies #-}
>
> module Main (main) where
>
> import Control.Arrow (arr)
> import Control.Exception (evaluate)
> import Control.Monad (forM_, when)
> import Control.Monad.Trans.State.Strict (get, put)
> import Data.Functor.Identity (Identity (..))
> import Data.IORef (newIORef, readIORef, writeIORef)
> import GHC.Generics (Generic)
> import GHC.TypeNats (KnownNat)
> import System.Environment (getArgs)
> import System.Random (mkStdGen, randomRs)
> import Text.Printf (printf)
>
> import qualified Synapse.Autograd as AG
> import qualified Synapse.NN as NN
> import qualified Synapse.Tensor as T

The model contains an LSTM with eight hidden units and a dense projection to one output value.
Its parameter record is independent of the recurrent state used while processing a sequence.
`TensorStructure` lets differentiation, optimization, and execution traverse the same record.

> data Parameters t = Parameters
>     { memory :: NN.LSTM 1 8 t Double
>     , output :: NN.Dense 8 1 t Double
>     }
>     deriving (Generic)
>
> instance T.TensorStructure (Parameters T.Tensor) where
>     type WithTensor t (Parameters T.Tensor) = Parameters t

The loader produces `[batch, time, features]`, whereas LSTM expects `[time, batch, features]`.
Transpose these axes before the recurrent layer and transpose the predictions back afterward.
The LSTM returns a hidden vector at every time step; dense applies the same projection to all of them.

`Forward` carries an `LSTMState` containing both hidden and cell tensors.
Unlike `BatchNorm`'s running statistics, these tensors participate in differentiation within the sequence.
The representation remains polymorphic, allowing the same forward definition in training and inference.

> predict
>     :: forall batch_size t
>      . (KnownNat batch_size, T.RealFloatTensorOps t Double)
>     => Parameters t
>     -> NN.Forward
>         (NN.LSTMState batch_size 8 t Double)
>         (t '[batch_size, 8, 1] Double)
>         (t '[batch_size, 8, 1] Double)
> predict parameters = proc input -> do
>     timeMajor <- arr (T.permuteAxes @'[1, 0, 2]) -< input

`lstm` already returns a `State` computation, so wrap it directly in `Forward`.
The other operations use their existing arrow instances or are lifted with `arr`.

>     hidden <- NN.Forward (NN.lstm $ memory parameters) -< timeMajor
>     projected <- NN.dense (output parameters) -< hidden
>     arr (T.permuteAxes @'[1, 0, 2]) -< projected

Generate reproducible input sequences and calculate their target moving averages.
Each sample is independent and begins with zero history.
Different seed ranges provide separate training and validation sequences.

> sample
>     :: Int
>     -> ([Double], [Double])
> sample index =
>     let inputs = take 8 $ randomRs (-0.8, 0.8) $ mkStdGen index
>         targets = drop 1 $ scanl (\previous input -> 0.7 * previous + 0.3 * input) 0 inputs
>      in (inputs, targets)

Each dataset sample contains an input tensor and a target tensor, both shaped `[8, 1]`.
Batching prepends the batch dimension without changing the time or feature axes.

> dataset
>     :: Int
>     -> Int
>     -> NN.Dataset Identity (T.TensorData '[8, 1] Double, T.TensorData '[8, 1] Double)
> dataset count offset = NN.Indexed count $ \index ->
>     let (inputs, targets) = sample $ index + offset
>      in pure (T.fromListData inputs, T.fromListData targets)

Initialize the input and recurrent projections with Xavier initialization, and biases with zero.
LSTM packs its four gates into the final dimension: 4 * 8 = 32 outputs in each projection.
The dense layer maps eight hidden features to one predicted signal value.

> initialParameters :: Parameters T.Tensor
> initialParameters = NN.evalState initialize $ mkStdGen 42
>   where
>     initialize = do
>         inputWeights <- NN.xavierUniform 1 32 1
>         recurrentWeights <- NN.xavierUniform 8 32 1
>         outputWeights <- NN.xavierUniform 8 1 1
>         pure $
>             Parameters
>                 (NN.LSTM inputWeights recurrentWeights 0 0)
>                 (NN.Dense outputWeights 0)

Adam retains its moments and step counter between batches.
Recurrent state has a different lifetime here: reset it for every independent batch of sequences.

> optimizer :: NN.Adam
> optimizer = NN.Adam 0.01 0.9 0.999 1e-8

Differentiate mean squared error over every sample and time step.
The zero recurrent state is inferred in the differentiable tensor representation.

Clip gradients by their global L2 norm before Adam updates the parameters.
When the norm exceeds one, all parameter gradients receive the same scale factor.

The training step updates its parameters and optimizer state through `State`, returning only the loss.
The batch executor uses `runState` to obtain the updated tensors at the execution boundary.

> type TrainingState = (Parameters T.Tensor, NN.OptimizerState NN.Adam (Parameters T.Tensor))
>
> trainingStep
>     :: KnownNat batch_size
>     => T.Tensor '[batch_size, 8, 1] Double
>     -> T.Tensor '[batch_size, 8, 1] Double
>     -> NN.State TrainingState (T.Tensor '[] Double)
> trainingStep features targets = do
>     (parameters, optimizerState) <- get
>     let (loss, gradients) =
>             AG.backprop
>                 ( \p ->
>                     let predictions =
>                             NN.evalState (NN.forward (predict p) $ AG.constant features) $ NN.LSTMState 0 0
>                      in NN.mse T.mean predictions $ AG.constant targets
>                 )
>                 parameters
>         clipped = NN.clipGradNormL2 1 gradients
>         (updated, nextOptimizer) = NN.runState (NN.optimizerStep optimizer parameters clipped) optimizerState
>     put (updated, nextOptimizer)
>     pure loss

Validation starts each sequence from zero state and compares all eight predictions with their targets.

> validationStep
>     :: KnownNat batch_size
>     => Parameters T.Tensor
>     -> T.Tensor '[batch_size, 8, 1] Double
>     -> T.Tensor '[batch_size, 8, 1] Double
>     -> T.Tensor '[] Double
> validationStep parameters features targets =
>     let predictions = NN.evalState (NN.forward (predict parameters) features) $ NN.LSTMState 0 0
>      in NN.mse T.mean predictions targets

The same learned parameters can process one input at a time through `lstmCell`.
For streaming inference, return the updated state and feed it into the next call.
This preserves the sequence history across executions instead of resetting it at every input.

> streamingStep
>     :: Parameters T.Tensor
>     -> NN.LSTMState 1 8 T.Tensor Double
>     -> T.Tensor '[1, 1] Double
>     -> (T.Tensor '[1, 1] Double, NN.LSTMState 1 8 T.Tensor Double)
> streamingStep parameters state input =
>     let (hidden, next) = NN.runState (NN.lstmCell (memory parameters) input) state
>      in (NN.dense (output parameters) hidden, next)

Synapse tensors describe a staged computation.
To execute it, we select a backend: CPU by default, or GPU when explicitly requested.
GPU execution requires Synapse to be built with the Cabal flag -fgpu and a supported CUDA device.

> chooseBackend :: IO T.Backend
> chooseBackend = do
>     args <- getArgs
>     case args of
>         [] -> pure T.defaultCPU
>         ["--cpu"] -> pure T.defaultCPU
>         ["--gpu"] -> pure T.defaultGPU
>         _ -> ioError $ userError "Usage: recurrent-example [--cpu | --gpu]"

Prepare one training executor for batches of 16 and one validation executor for batches of 32.
Both datasets divide evenly into their batch sizes.
Retain the prepared functions across epochs to reuse their computation graphs.

> main :: IO ()
> main = do
>     backend <- chooseBackend
>     let training = dataset 128 0
>         validation = dataset 32 1000
>         trainingPlan = NN.planBatches 16 NN.KeepLast training
>         validationPlan = NN.planBatches 32 NN.KeepLast validation
>         step = NN.prepareBatchStep backend trainingPlan $ \s x y _ -> NN.runState (trainingStep x y) s
>         measure = NN.prepareBatchStep backend validationPlan $ \p x y _ -> validationStep p x y

Materialize the model parameters and initial Adam state together.
A separate generator supplies a new sample order for each training epoch.

>     model <- newIORef $ T.run backend (initialParameters, NN.initialState optimizer initialParameters)
>     generator <- newIORef $ mkStdGen 42
>
>     putStrLn "Moving average: LSTM (8 hidden units) -> Dense; Adam."
>     putStrLn "Independent sequences of 8 steps; recurrent state resets for each batch."

Epoch zero measures the initial model.
For later epochs, shuffle the dataset and load batches using the same sizes as the preparation plan.

>     forM_ [0 .. 100 :: Int] $ \epoch -> do
>         when (epoch > 0) $ do
>             random <- readIORef generator
>             let (source, nextGenerator) = NN.runState (NN.shuffle training) random
>             writeIORef generator nextGenerator

Each step retains the updated parameters and optimizer state.
The recurrent state resets inside `trainingStep`,
because rows in the next batch represent unrelated sequences.
Force the scalar loss to execute each update before retaining its results.

>             forM_ (NN.loadBatches 16 NN.KeepLast source) $ \(Identity batch) -> do
>                 current <- readIORef model
>                 (loss, updated) <- either (fail . show) pure $ NN.runBatch step current batch
>                 _ <- evaluate $ sum $ T.toList loss
>                 writeIORef model updated

Every 20 epochs, measure mean squared error on the held-out sequences.

>         when (epoch `mod` 20 == 0) $
>             forM_ (NN.batchLoads validationPlan) $ \(Identity batch) -> do
>                 (parameters, _) <- readIORef model
>                 loss <- either (fail . show) pure $ NN.runBatch measure parameters batch
>                 printf "Epoch %3d | validation MSE %.6f\n" epoch (sum $ T.toList loss)

After training, prepare a single-cell execution function for a batch containing one stream.
Initialize its recurrent state once, then retain the next state after every input.
The table compares streaming predictions with the target moving average on a new sequence.

>     (parameters, _) <- readIORef model
>     let (inputs, targets) = sample 2000
>         cell = T.run backend streamingStep
>     recurrentState <- newIORef $ T.run backend (NN.LSTMState 0 0 :: NN.LSTMState 1 8 T.Tensor Double)
>     putStrLn "\nStreaming inference on a new sequence:"
>     putStrLn " input   target  predicted"
>     forM_ (zip inputs targets) $ \(input, target) -> do
>         state <- readIORef recurrentState
>         let (prediction, nextState) = cell parameters state $ T.fromListData @'[1, 1] [input]
>         _ <- evaluate $ sum $ T.toList prediction
>         writeIORef recurrentState nextState
>         printf "% .3f  % .3f  % .3f\n" input target (sum $ T.toList prediction)
