Predicting repeated motifs with a Transformer
===========================================

Predict the next token in a repeating three-token motif drawn from a vocabulary of eight tokens.
For example, [2, 5, 1, 2, 5, 1] has next-token targets [5, 1, 2, 5, 1, 2].
A causal Transformer must use the observed prefix to identify the repetition without seeing future tokens.

> {-# LANGUAGE Arrows #-}
> {-# LANGUAGE ImpredicativeTypes #-}
> {-# LANGUAGE TypeFamilies #-}
>
> module Main (main) where
>
> import Control.Arrow (Arrow, arr, (>>>))
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

Use one Transformer block with feature width eight and two attention heads.
Each head has depth four.
The feed-forward sublayer expands eight features to sixteen and projects back to eight.
Both sublayers use `LayerNorm` before their computation and a residual connection afterward.

The embedding table will also project final features to vocabulary logits.
There is one shared parameter tensor for both roles.

> data Parameters t = Parameters
>     { tokens :: NN.Embedding 8 8 t Double
>     , selfAttention :: NN.MultiHeadAttention 8 2 t Double
>     , attentionNorm :: NN.LayerNorm '[8] t Double
>     , feedForwardNorm :: NN.LayerNorm '[8] t Double
>     , expand :: NN.Dense 8 16 t Double
>     , project :: NN.Dense 16 8 t Double
>     }
>     deriving (Generic)
>
> instance T.TensorStructure (Parameters T.Tensor) where
>     type WithTensor t (Parameters T.Tensor) = Parameters t

Build attention from its projections so rotary position encoding can act on queries and keys.
The input has shape `[batch, 6, 8]`; `splitHeads` produces `[batch, 2, 6, 4]`.
All three projections are learned, while position encoding has no trainable parameters.

> rotaryAttention
>     :: forall batch_size t arrow
>      . (KnownNat batch_size, Arrow arrow, T.RealFloatTensorOps t Double)
>     => NN.MultiHeadAttention 8 2 t Double
>     -> arrow (t '[batch_size, 6, 8] Double) (t '[batch_size, 6, 8] Double)
> rotaryAttention parameters = proc input -> do
>     projectedQ <- NN.dense (NN.attentionQuery parameters) -< input
>     projectedK <- NN.dense (NN.attentionKey parameters) -< input
>     projectedV <- NN.dense (NN.attentionValue parameters) -< input

Apply RoPE independently within each head, leaving values unchanged.
Offset zero starts positions at the beginning of each independent sequence.
The broadcast causal mask permits keys at the same or earlier position, including the diagonal.
Attention computes weighted values; `mergeHeads` rejoins the heads before the learned output projection.

>     let query = NN.rope 10000 0 $ NN.splitHeads @2 projectedQ
>         key = NN.rope 10000 0 $ NN.splitHeads @2 projectedK
>         value = NN.splitHeads @2 projectedV
>         mask = T.broadcast $ NN.causalMask @6 @6
>         attended = NN.attention (Just mask) query key value
>     NN.dense (NN.attentionOutput parameters) -< NN.mergeHeads attended

Token indices remain ordinary integer tensors during differentiation.
The embedding operation gathers their learned vectors, so gradients update the table rather than the indices.

> predict
>     :: (KnownNat batch_size, Arrow arrow, T.RealFloatTensorOps t Double)
>     => Parameters t
>     -> arrow (T.Tensor '[batch_size, 6] Int) (t '[batch_size, 6, 8] Double)
> predict parameters = proc input -> do
>     embedded <- NN.embedding (tokens parameters) -< input
>     normalized <- NN.layerNorm 1e-5 (attentionNorm parameters) -< embedded
>     attended <- rotaryAttention (selfAttention parameters) -< normalized

Retain the original embedding values and add the attention output to form the first residual connection.
Normalize the result before the feed-forward sublayer, then add that sublayer's output as a second residual.

>     let residual = embedded + attended
>     normalizedAgain <- NN.layerNorm 1e-5 (feedForwardNorm parameters) -< residual
>     expanded <- NN.dense (expand parameters) >>> NN.tanh -< normalizedAgain
>     projected <- NN.dense (project parameters) -< expanded

Transpose the embedding table from `[8, 8]` in vocabulary-by-feature order to feature-by-vocabulary order.
Although both dimensions happen to be eight here, transposition still changes the projection.
Using the same tensor for lookup and output projection accumulates gradients from both uses into one parameter.
The result has shape `[batch, 6, 8]`: one logit per vocabulary token at each position.

>     arr (\x -> T.matmul x $ T.transpose $ NN.embeddingWeights $ tokens parameters) -< residual + projected

Encode each of the 512 possible three-token motifs as an integer.
Multiplication by 73 modulo 512 permutes those codes without repetition.
Use disjoint index ranges for training and validation, so validation motifs were not seen during training.

> sample
>     :: Int
>     -> ([Int], [Int])
> sample index =
>     let code = (73 * index) `mod` 512
>         motif = [code `div` 64, (code `div` 8) `mod` 8, code `mod` 8]
>         sequenceTokens = cycle motif
>      in (take 6 sequenceTokens, take 6 $ drop 1 sequenceTokens)

Each sample is a pair of six-token integer tensors.
Use the same Dataset and batching API as the other neural-network examples.

> dataset
>     :: Int
>     -> Int
>     -> NN.Dataset Identity (T.TensorData '[6] Int, T.TensorData '[6] Int)
> dataset count offset = NN.Indexed count $ \index ->
>     let (inputs, targets) = sample $ index + offset
>      in pure (T.fromListData @'[6] inputs, T.fromListData @'[6] targets)

Initialize the embedding table with small normal values and the dense projections with Xavier initialization.
`LayerNorm` scales start at one and biases at zero.
No separate output classifier is initialized, because it reuses the embedding weights.

> initialParameters :: Parameters T.Tensor
> initialParameters = NN.evalState initialize $ mkStdGen 7
>   where
>     initialize = do
>         embedding <- NN.normal 0 0.2
>         query <- NN.xavierUniform 8 8 1
>         key <- NN.xavierUniform 8 8 1
>         value <- NN.xavierUniform 8 8 1
>         output <- NN.xavierUniform 8 8 1
>         expandWeights <- NN.xavierUniform 8 16 1
>         projectWeights <- NN.xavierUniform 16 8 1
>         pure $
>             Parameters
>                 (NN.Embedding embedding)
>                 ( NN.MultiHeadAttention
>                     (NN.Dense query 0)
>                     (NN.Dense key 0)
>                     (NN.Dense value 0)
>                     (NN.Dense output 0)
>                 )
>                 (NN.LayerNorm 1 0)
>                 (NN.LayerNorm 1 0)
>                 (NN.Dense expandWeights 0)
>                 (NN.Dense projectWeights 0)

Use AdamW with decoupled weight decay.
Its moments and step counter persist between updates, independently of the current input sequences.

> optimizer :: NN.AdamW
> optimizer = NN.AdamW (NN.Adam 0.01 0.9 0.999 1e-8) 0.01

The first two next-token targets are not generally determined by the observed prefix.
Once three tokens have been observed, subsequent targets follow from the repeated motif.
Train only on positions 2 through 5 by slicing both logits and labels before cross entropy.

The class axis is 2, following batch and time.
Targets are known to lie in `[0, 7]`, satisfying the unchecked index requirement of `unsafeCrossEntropy`.

The training step updates its parameters and optimizer state through `State`, returning only the loss.
The batch executor uses `runState` to obtain the updated tensors at the execution boundary.

> type TrainingState = (Parameters T.Tensor, NN.OptimizerState NN.AdamW (Parameters T.Tensor))
>
> trainingStep
>     :: KnownNat batch_size
>     => T.Tensor '[batch_size, 6] Int
>     -> T.Tensor '[batch_size, 6] Int
>     -> NN.State TrainingState (T.Tensor '[] Double)
> trainingStep features targets = do
>     (parameters, optimizerState) <- get
>     let (loss, gradients) =
>             AG.backprop
>                 ( \p ->
>                     NN.unsafeCrossEntropy @2
>                         T.mean
>                         (T.slice @'[T.All, T.Range 2 6 1, T.All] $ predict p features)
>                         (T.slice @'[T.All, T.Range 2 6 1] targets)
>                 )
>                 parameters
>         updated = NN.runState (NN.optimizerStep optimizer parameters gradients) optimizerState
>     put updated
>     pure loss

Measure loss and token accuracy on the same four positions of each held-out sequence.
Accuracy compares the largest logit with the target index; no softmax is needed.

> validationStep
>     :: KnownNat batch_size
>     => Parameters T.Tensor
>     -> T.Tensor '[batch_size, 6] Int
>     -> T.Tensor '[batch_size, 6] Int
>     -> (T.Tensor '[] Double, T.Tensor '[] Double)
> validationStep parameters features targets =
>     let logits = T.slice @'[T.All, T.Range 2 6 1, T.All] $ predict parameters features
>         labels = T.slice @'[T.All, T.Range 2 6 1] targets
>      in (NN.unsafeCrossEntropy @2 T.mean logits labels, NN.accuracy @2 logits labels)

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
>         _ -> ioError $ userError "Usage: transformer-example [--cpu | --gpu]"

The 128 training motifs fit in one batch; validation uses 32 different motifs.
Prepare both executors before the loop and retain them across epochs.
The training adapter runs the `State` computation and returns its loss and updated state to the executor.

> main :: IO ()
> main = do
>     backend <- chooseBackend
>     let training = dataset 128 0
>         validation = dataset 32 400
>         trainingPlan = NN.planBatches 128 NN.KeepLast training
>         validationPlan = NN.planBatches 32 NN.KeepLast validation
>         step = NN.prepareBatchStep backend trainingPlan $ \s x y _ -> NN.runState (trainingStep x y) s
>         measure = NN.prepareBatchStep backend validationPlan $ \p x y _ -> validationStep p x y

Materialize the parameters and AdamW state together.
Retain their latest values between batches and keep a generator for shuffling the training motifs.

>     model <- newIORef $ T.run backend (initialParameters, NN.initialState optimizer initialParameters)
>     generator <- newIORef $ mkStdGen 42
>
>     putStrLn "Next-token prediction: repeat a three-token motif from an eight-token vocabulary."
>     putStrLn "Embedding -> causal attention with RoPE -> MLP; pre-norm, residuals, tied weights, AdamW."

Epoch zero measures the initial model.
For subsequent epochs, shuffle the motifs and retain the updated random generator.
Batch sizes stay unchanged, so the same executors remain applicable.

>     forM_ [0 .. 200 :: Int] $ \epoch -> do
>         when (epoch > 0) $ do
>             random <- readIORef generator
>             let (source, nextGenerator) = NN.runState (NN.shuffle training) random
>             writeIORef generator nextGenerator

Load each batch and run the prepared training step with the current parameters and optimizer state.
Force the loss so the update executes before retaining the resulting tensors.

>             forM_ (NN.loadBatches 128 NN.KeepLast source) $ \(Identity batch) -> do
>                 current <- readIORef model
>                 (loss, updated) <- either (fail . show) pure $ NN.runBatch step current batch
>                 _ <- evaluate $ sum $ T.toList loss
>                 writeIORef model updated

Every 25 epochs, report validation loss and accuracy on positions 2 through 5.
Validation needs the current parameters without the optimizer state.

>         when (epoch `mod` 25 == 0) $
>             forM_ (NN.batchLoads validationPlan) $ \(Identity batch) -> do
>                 (parameters, _) <- readIORef model
>                 (loss, accuracy) <- either (fail . show) pure $ NN.runBatch measure parameters batch
>                 printf
>                     "Epoch %3d | validation loss %.4f | accuracy %.1f%%\n"
>                     epoch
>                     (sum $ T.toList loss)
>                     (100 * sum (T.toList accuracy))

Finally, prepare inference for one sequence and reuse it for several held-out examples.
Print predictions only after the first three tokens have been observed, matching the training objective.

>     (trained, _) <- readIORef model
>     let infer = T.run backend $ \p x -> T.argmax @2 $ predict p x
>     putStrLn "\nHeld-out motifs (next-token predictions after observing the first three tokens):"
>     forM_ [400, 407, 419, 431] $ \index -> do
>         let (input, labels) = sample index
>             predicted = infer trained $ T.fromListData @'[1, 6] input
>         putStrLn $ "Input:     " ++ show input
>         putStrLn $ "Expected:  " ++ show (drop 2 labels)
>         putStrLn $ "Predicted: " ++ show (drop 2 $ T.toList predicted)
