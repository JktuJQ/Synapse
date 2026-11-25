Recognizing bars with a residual convolutional network
=====================================================

Classify noisy 8 by 8 images containing either a horizontal or a vertical bar.
The bar can appear at different positions, so the model must recognize its orientation.
Use convolutions to extract local features and a residual block to refine them.

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
> import System.Random (mkStdGen)
> import Text.Printf (printf)
>
> import qualified Synapse.Autograd as AG
> import qualified Synapse.NN as NN
> import qualified Synapse.Tensor as T

Store trainable parameters in a record, independently of the forward computation.
The first convolution maps one input channel to four feature channels.
The residual branch contains two convolutions, each followed by `BatchNorm`.
Each normalization has a learned per-channel scale and bias.
Pooling reduces the feature maps to 4 by 4, leaving 64 features for the final dense layer.

> data Parameters t = Parameters
>     { stem :: NN.Convolution 1 4 '[3, 3] 1 t Double
>     , residualFirst :: NN.Convolution 4 4 '[3, 3] 1 t Double
>     , firstNorm :: NN.BatchNorm 4 t Double
>     , residualSecond :: NN.Convolution 4 4 '[3, 3] 1 t Double
>     , secondNorm :: NN.BatchNorm 4 t Double
>     , output :: NN.Dense 64 2 t Double
>     }
>     deriving (Generic)
>
> instance T.TensorStructure (Parameters T.Tensor) where
>     type WithTensor t (Parameters T.Tensor) = Parameters t

`BatchNorm` also keeps running means and variances for evaluation.
These statistics belong to model state rather than trainable parameters.
Both records implement `TensorStructure`, so they can pass through backend execution together.

> data ModelState t = ModelState
>     { firstStatistics :: NN.BatchNormState 4 t Double
>     , secondStatistics :: NN.BatchNormState 4 t Double
>     }
>     deriving (Generic)
>
> instance T.TensorStructure (ModelState T.Tensor) where
>     type WithTensor t (ModelState T.Tensor) = ModelState t

For the most convenient way of working with stateful layers we will need lenses.

> firstStatisticsLens
>     :: Functor f
>     => (NN.BatchNormState 4 t Double -> f (NN.BatchNormState 4 t Double))
>     -> ModelState t
>     -> f (ModelState t)
> firstStatisticsLens change state =
>     fmap (\updated -> state{firstStatistics = updated}) $ change $ firstStatistics state
>
> secondStatisticsLens
>     :: Functor f
>     => (NN.BatchNormState 4 t Double -> f (NN.BatchNormState 4 t Double))
>     -> ModelState t
>     -> f (ModelState t)
> secondStatisticsLens change state =
>     fmap (\updated -> state{secondStatistics = updated}) $ change $ secondStatistics state

The forward pass is a `Forward` arrow: it produces logits while threading model state through its layers.

> predict
>     :: forall batch_size t
>      . ( KnownNat batch_size
>        , T.RealFloatTensorOps t Double
>        )
>     => NN.ForwardMode
>     -> Parameters t
>     -> NN.Forward (ModelState T.Tensor) (t '[batch_size, 1, 8, 8] Double) (t '[batch_size, 2] Double)
> predict mode parameters = proc input -> do
>     embedded <- NN.conv2d @'[1, 1] @'[1, 1, 1, 1] @'[1, 1] (stem parameters) -< input
>     shortcut <- NN.relu -< embedded

All convolutions use stride one, padding one on every side, and dilation one.
The residual branch therefore preserves the shape `[batch_size, 4, 8, 8]`.
`stateful` uses each lens to route `BatchNorm`'s local state through the enclosing `ModelState`.
In `Train` mode, normalization uses batch statistics and updates the running estimates;
in `Eval` mode, it uses the stored estimates and preserves them.

>     first <- NN.conv2d @'[1, 1] @'[1, 1, 1, 1] @'[1, 1] (residualFirst parameters) -< shortcut
>     normalized <-
>         NN.stateful firstStatisticsLens (NN.batchNorm @1 mode 1e-5 0.1 $ firstNorm parameters) -< first
>     activated <- NN.relu -< normalized
>     second <- NN.conv2d @'[1, 1] @'[1, 1, 1, 1] @'[1, 1] (residualSecond parameters) -< activated
>     refined <-
>         NN.stateful secondStatisticsLens (NN.batchNorm @1 mode 1e-5 0.1 $ secondNorm parameters) -< second

Add the original features to the branch output before applying ReLU.
The skip connection is simply another use of `shortcut`; flow of computation is separate from data.

>     combined <- NN.relu -< shortcut + refined
>     flattened <-
>         arr (T.reshape @'[batch_size, 64] . T.maxPool2d @'[2, 2] @'[2, 2] @'[0, 0, 0, 0] @'[1, 1]) -< combined
>     NN.dense (output parameters) -< flattened

Generate horizontal bars with label 0 and vertical bars with label 1.
Each image has one channel; each target is a scalar integer class index.
Changing the offset produces different noise patterns for validation.

> dataset
>     :: Int
>     -> Int
>     -> NN.Dataset Identity (T.TensorData '[1, 8, 8] Double, T.TensorData '[] Int)
> dataset count offset = NN.Indexed count $ \index ->
>     let sample = index + offset
>         label = sample `mod` 2
>         position = 1 + (sample `div` 2) `mod` 6
>         pixel row column =
>             let coordinate = if label == 0 then row else column
>                 bar = if coordinate == position || coordinate == position + 1 then 1 else 0
>                 noise = 0.08 * sin (fromIntegral $ sample * 17 + row * 11 + column * 7)
>              in bar + noise
>      in pure
>             ( T.fromListData @'[1, 8, 8] [pixel row column | row <- [0 .. 7], column <- [0 .. 7]]
>             , T.fromListData @'[] [label]
>             )

Initialize convolution weights with Kaiming initialization and the dense weights with Xavier initialization.
Convolution and dense biases start at zero; `BatchNorm` scales start at one and biases at zero.
The fan-in is 9 for the first convolution and 36 for each four-channel convolution.

> initialParameters :: Parameters T.Tensor
> initialParameters = NN.evalState initialize $ mkStdGen 42
>   where
>     initialize = do
>         stemWeights <- NN.kaimingNormal @'[4, 1, 3, 3] 9 0
>         firstWeights <- NN.kaimingNormal @'[4, 4, 3, 3] 36 0
>         secondWeights <- NN.kaimingNormal @'[4, 4, 3, 3] 36 0
>         outputWeights <- NN.xavierUniform @'[64, 2] 64 2 1
>         pure $
>             Parameters
>                 (NN.Convolution stemWeights 0)
>                 (NN.Convolution firstWeights 0)
>                 (NN.BatchNorm 1 0)
>                 (NN.Convolution secondWeights 0)
>                 (NN.BatchNorm 1 0)
>                 (NN.Dense outputWeights 0)
>
> initialModelState :: ModelState T.Tensor
> initialModelState = ModelState NN.initialBatchNormState NN.initialBatchNormState
>
> optimizer :: NN.RMSprop
> optimizer = NN.RMSprop 0.003 0.9 1e-8

`forward` returns a `State` computation; `runState` extracts the logits and updated model statistics.

The training step carries parameters, model statistics, and optimizer state in one `State` computation.
It returns only the loss; `runState` exposes the updated tensors at the execution boundary.

> type TrainingState =
>     ( Parameters T.Tensor
>     , ModelState T.Tensor
>     , NN.OptimizerState NN.RMSprop (Parameters T.Tensor)
>     )
>
> trainingStep
>     :: KnownNat batch_size
>     => T.Tensor '[batch_size, 1, 8, 8] Double
>     -> T.Tensor '[batch_size] Int
>     -> NN.State TrainingState (T.Tensor '[] Double)
> trainingStep features targets = do
>     (parameters, state, optimizerState) <- get
>     let ((loss, nextModel), gradients) =
>             AG.backprop
>                 ( \p ->
>                     let (logits, next) =
>                             NN.runState (NN.forward (predict NN.Train p) $ AG.constant features) state
>                      in (NN.unsafeCrossEntropy @1 T.mean logits targets, AG.constant next)
>                 )
>                 parameters
>         (updated, nextOptimizer) =
>             NN.runState (NN.optimizerStep optimizer parameters gradients) optimizerState
>     put (updated, nextModel, nextOptimizer)
>     pure loss

For validation, use `Eval` mode and the running statistics learned during training.
`evalState` extracts the logits; no statistics or optimizer state need to be updated.
The final layer returns logits directly, suitable for both cross entropy and accuracy.

> validationStep
>     :: KnownNat batch_size
>     => Parameters T.Tensor
>     -> ModelState T.Tensor
>     -> T.Tensor '[batch_size, 1, 8, 8] Double
>     -> T.Tensor '[batch_size] Int
>     -> (T.Tensor '[] Double, T.Tensor '[] Double)
> validationStep parameters state features targets =
>     let logits = NN.evalState (NN.forward (predict NN.Eval parameters) features) state
>      in (NN.unsafeCrossEntropy @1 T.mean logits targets, NN.accuracy @1 logits targets)

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
>         _ -> ioError $ userError "Usage: convolution-example [--cpu | --gpu]"

Prepare training and validation executors before entering the epoch loop.
The training set divides evenly into batches of 16; validation uses a single batch of 32.
Retain the executors across epochs to reuse their prepared computations.

> main :: IO ()
> main = do
>     backend <- chooseBackend
>     let training = dataset 128 0
>         validation = dataset 32 1000
>         trainingPlan = NN.planBatches 16 NN.KeepLast training
>         validationPlan = NN.planBatches 32 NN.KeepLast validation
>         step = NN.prepareBatchStep backend trainingPlan $ \s x y _ -> NN.runState (trainingStep x y) s
>         measure = NN.prepareBatchStep backend validationPlan $ \(p, s) x y _ -> validationStep p s x y

Materialize the parameters, `BatchNorm` statistics, and `RMSprop` state together.
Retain their updated values between batches, along with a generator for shuffling the training data.

>     model <-
>         newIORef $
>             T.run backend
>                 (initialParameters, initialModelState, NN.initialState optimizer initialParameters)
>     generator <- newIORef $ mkStdGen 42
>
>     putStrLn "Horizontal vs vertical bars: Conv 1 -> 4 -> residual block -> Pool -> Dense 64 -> 2 logits."

At each training epoch, shuffle the samples and retain the updated generator.
The dimensions stay the same after shuffling, so the training executor remains applicable.

>     forM_ [0 .. 50 :: Int] $ \epoch -> do
>         when (epoch > 0) $ do
>             random <- readIORef generator
>             let (source, nextGenerator) = NN.runState (NN.shuffle training) random
>             writeIORef generator nextGenerator

Run the prepared training step for each loaded batch, carrying all three parts of the model forward.
Force the loss to execute the computation before retaining the updated tensors.

>             forM_ (NN.loadBatches 16 NN.KeepLast source) $ \(Identity batch) -> do
>                 current <- readIORef model
>                 (loss, updated) <- either (fail . show) pure $ NN.runBatch step current batch
>                 _ <- evaluate $ sum $ T.toList loss
>                 writeIORef model updated

Epoch zero reports the untrained model; every ten epochs, measure validation loss and accuracy.
Only parameters and `BatchNorm` statistics are passed to validation.

>         when (epoch `mod` 10 == 0) $
>             forM_ (NN.batchLoads validationPlan) $ \(Identity batch) -> do
>                 (parameters, state, _) <- readIORef model
>                 (loss, accuracy) <- either (fail . show) pure $ NN.runBatch measure (parameters, state) batch
>                 printf
>                     "Epoch %2d | loss %.4f | accuracy %.1f%%\n"
>                     epoch
>                     (sum $ T.toList loss)
>                     (100 * sum (T.toList accuracy))
