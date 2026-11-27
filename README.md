<div align="center">
  <h1><img src="SynapseLogo.png" width="160" alt="Synapse logo"><br>Synapse</h1>
</div>

`Synapse` is a machine learning library for Haskell.
It provides tensors, automatic differentiation, and tools for building and training neural networks.

## 🔨 Design 🔨

Haskell has a handful of neural network libraries, each with its own approach.
The [`neural`][neural] library composes models and uses pipes to assemble training loops.
[`Grenade`][grenade] uses the type system to check that layers fit together.
[`Hasktorch`][hasktorch] builds on LibTorch and brings its tensors and neural network operations to Haskell.

`Synapse` brings those ideas together, while still remaining easily extensible, simple and powerful:

* `Arrow`-based composition
* Type-level shapes and checks
* Backend-accelerated execution

**Interested?** Let's dive right in!

## 💻 Usage 💻

`Synapse` comes batteries-included with
its own tensors,
autodifferentiation system,
and neural networks building blocks.

### Tensors

`Synapse.Tensor` provides tensors and the operations used to build neural network computations.
Its familiar operations work like those in Torch,
with tensor shapes checked at compile time.

For example, adding tensors with incompatible shapes or
reducing along an axis that does not exist is a type error.
The program will not compile until the shapes and axes are valid.

```haskell
left  :: T.Tensor '[2, 3] Double
right :: T.Tensor '[2, 3] Double
valid = left + right

-- This does not compile: the shapes differ.
right' :: T.Tensor '[3, 2] Double
invalid = left + right'
```

Operations such as `broadcast` and `reshape` make shape changes explicit.
They let you express the same ideas as in Torch while keeping each intermediate
shape visible to the type checker.

Here is a small linear layer written directly from tensor operations:

```haskell
linear
    :: T.Tensor '[batch, input] Double
    -> T.Tensor '[input, output] Double
    -> T.Tensor '[output] Double
    -> T.Tensor '[batch, output] Double
linear input weights bias =
    input `T.matmul` weights + T.broadcast bias
```

Synapse uses fabulous [Accelerate][accelerate] library as its computation backend.
It provides a powerful DSL to express computation, and,
if `Synapse` does not provide some operation you might want,
you could just use the escape hatch to write your own.

Tensor operations are staged into a computation graph,
which is compiled for the selected backend when it is run
(which could be done just-in-time, ahead-of-time, or even asynchronously).
This allows the same tensor code to run on different supported architectures,
including GPUs when `Synapse` is built with GPU support.

```haskell
expression :: T.Tensor '[2] Double

dataResult :: T.TensorData '[2] Double
dataResult = T.run T.defaultCPU expression

step :: T.Tensor '[2] Double -> T.Tensor '[2] Double
step = (* 2)

runStep :: T.TensorData '[2] Double -> T.TensorData '[2] Double
runStep = T.run T.defaultCPU step
```

`run` executes a closed expression and returns materialized `TensorData`.
When given a function,
`run` prepares its staged computation and returns a function that accepts and returns materialized data.
Keep the prepared function and reuse it for later evaluations.
This retains the compiled computation and lets the backend reuse its cache.

For matrix multiplication,
Synapse can use BLAS on the CPU and cuBLAS on the GPU when built with the `blas` flag.
The generated backend code enables high-performance execution (see [benchmarks](#benchmarks)).

<p>
<sub>
If you worked with BLAS, you know that it provides optimized matrix products with transpose options.
You don't need to remember selecting the correct option:
`Synapse`'s rewrite rules recognize transpose-and-multiply patterns and select those automatically,
so ordinary `matmul` expressions can use the optimized path.
</sub>
</p>

See the [Tensor example][tensor-example] written in literate Haskell for a complete worked example.

### Automatic differentiation

`Synapse.Autograd` computes gradients with reverse-mode automatic differentiation.
It uses [backprop][backprop] library to track tensor variables and accumulate gradient contributions.
You write the function you want to differentiate, and Synapse returns its value and gradient:

```haskell
input = 3 :: AG.Tensor '[] Double
(value, derivative) = AG.backprop (\x -> x * x) input

T.toList $ T.run T.defaultCPU value       -- [9.0]
T.toList $ T.run T.defaultCPU derivative  -- [6.0]
```

`T.Tensor` is not differentiable; `AG.Tensor` is.
But it still works with the operations from `Synapse.Tensor`.
Functions written polymorphically through `TensorOps` can work with each one:

```haskell
linear
    :: NumTensorOps t Double
    => t '[batch, input] Double
    -> t '[input, output] Double
    -> t '[output] Double
    -> t '[batch, output] Double
linear input weights bias =
    input `T.matmul` weights + T.broadcast bias
```

The same function can now be called with `T.Tensor` values for ordinary computation
or `AG.Tensor s` values while differentiating.
This pattern is heavily used by neural network layers, which we will cover below.

All polymorphic operations already have reverse rules.
When you define a custom operation, you can provide its vector-Jacobian product (VJP) with `customVJP`,
or use `withVJP` and `withCachedVJP` to adapt a forward function and its reverse rule.
This is similar to [JAX's custom VJP rules][jax-custom-vjp].

See the [Autograd example][autograd-example] written in literate Haskell for a complete worked example..

### Neural networks

#### Parameters, computation, and state

A model keeps its parameters, computation, and state separate.
Parameters are data that carries tensors learned during training.
Model state is data too, but it carries values between forward passes when a layer needs memory.
Computation is a function that describes how parameters and inputs produce outputs.

This distinction allows to leverage very neat Haskell patterns.

#### Parameters

Represent parameters as a regular Haskell type.
This example includes an embedding table, a normalization layer, and dense layers:

At this point, the record only describes values.
It does not say which layer runs first, how the layers are connected,
or whether one layer's output shape matches the next layer's input shape.
Those relationships appear in the computation we define separately.

```haskell
data Parameters t = Parameters
    { tokens :: NN.Embedding 8 8 t Double
    , attentionNorm :: NN.LayerNorm '[8] t Double
    , expand :: NN.Dense 8 16 t Double
    , project :: NN.Dense 16 8 t Double
    }
    deriving (Generic)

instance T.TensorStructure (Parameters T.Tensor) where
    type WithTensor t (Parameters T.Tensor) = Parameters t
```

`Generic` and `TensorStructure` describe which tensors make up the parameters.
The `WithTensor` equation tells Synapse how to replace each tensor with another representation.
During differentiation, these tensors are registered as inputs to backpropagation,
and their gradients have the same record structure.

Initialize the parameters with suitable initializers.
Here, Xavier initialization sets the embedding and dense weights, while the biases start at zero:

```haskell
initialParameters = NN.evalState initialize $ mkStdGen 42
  where
    initialize = do
        embeddingWeights <- NN.xavierUniform @'[8, 8] 8 8 1
        hiddenWeights <- NN.xavierUniform @'[8, 16] 8 16 1
        projectWeights <- NN.xavierUniform @'[16, 8] 16 8 1
        pure $ Parameters
            (NN.Embedding embeddingWeights)
            (NN.LayerNorm 1 0)
            (NN.Dense hiddenWeights $ T.zeros @'[16])
            (NN.Dense projectWeights $ T.zeros @'[8])
```

#### Computation

The model computation is a function from parameters to a function from input to output.
You can describe the flow with ordinary function composition (right-to-left):

```haskell
predict parameters =
    NN.dense (output parameters)
        . NN.tanh
        . NN.dense (hidden parameters)
```

Or compose the same operations with `Arrow` (left-to-right):

```haskell
predict parameters =
    NN.dense (hidden parameters)
        >>> NN.dense (output parameters)
        >>> NN.tanh
```

`Arrow`s also allow to express a more complex computational flow.
We could reuse the same parameter, or create a skip-connection:

```haskell
predict parameters = proc input -> do
    embedded <- NN.embedding (tokens parameters) -< input
    attended <- rotaryAttention (selfAttention parameters) -< embedded
    let residual = embedded + attended
    expanded <- NN.tanh . NN.dense (expand parameters) -< residual
    projected <- NN.dense (project parameters) -< expanded
    let hidden = residual + projected
        logits = T.matmul hidden $ T.transpose $ NN.embeddingWeights (tokens parameters)
    returnA -< logits
```

Due to polymorphism, all functions work with `T.Tensor` or with `AG.Tensor`.

#### State

Some layers carry information between calls.
`Forward state input output` is an arrow from `input` to `output`
that threads a `state` through the computation.
Here is `Forward` definition:

```haskell
newtype Forward state input output = Forward
    { forward :: input -> State state output
    }
    deriving (Category, Arrow, ArrowChoice) via Kleisli (State state)
```

This lets stateless and stateful layers compose in one forward computation.
A model can collect the state of several layers in one record:

```haskell
data ModelState t = ModelState
    { firstStatistics :: NN.BatchNormState 4 t Double
    , secondStatistics :: NN.BatchNormState 4 t Double
    }
    deriving (Generic)
```

Each `BatchNorm` layer owns its running statistics.
A van-Laarhoven lens selects that part of `ModelState`,
and `stateful` lifts the layer to work with the whole state:

```haskell
normalized <-
    NN.stateful firstStatisticsLens
        (NN.batchNorm @1 mode 1e-5 0.1 (firstNorm parameters))
        -< features
```

The layer updates its own statistics and preserves the other fields.

#### Batching

The `Dataset` type describes a finite collection of examples and how to read them.
Each sample contains one input and one target.
which is useful when storage supports batched reads.
Both work in a caller-selected context such as `Identity` for in-memory data or `IO` for files.

Batching stacks samples and adds a leading dimension.
For example, inputs of shape `'[8]` become `'[32, 8]` and scalar targets become `'[32]`.
A training step makes this dimension explicit in its type:

```haskell
trainingStep
    :: KnownNat batch
    => T.Tensor '[batch, 8] Double
    -> T.Tensor '[batch] Int
    -> NN.State TrainingState (T.Tensor '[] Double)
```

`loadBatches` creates lazy loading actions, so the loop can load and consume one batch at a time:

```haskell
forM_ (NN.loadBatches 32 NN.KeepLast dataset) $ \load -> do
    someBatch <- load
    NN.withBatch someBatch $ \batch -> do
        let inputs = NN.batchInputs batch
            targets = NN.batchTargets batch
            mask = NN.batchMask batch
        trainOneBatch inputs targets mask
```

`LastBatch` controls what happens when the dataset size is not divisible by the requested batch size:

* `KeepLast` uses a smaller final batch.
* `DropLast` discards the remainder, keeping one batch size throughout.
* `PadLast input target` fills the final batch to the requested size and marks the real examples in `batchMask`.

`KeepLast` results in two different leading batch dimensions.
That will require a separately prepared specialization (because size is baked in computational graph),
although backend will try to reuse cache as much as possible.

`PadLast` requires masking; masked reductions add work to the computation,
so when practical, choose a batch size that divides the dataset and avoid padding.
The training step must apply it, for example with `maskedMean`.

Due to those remarks, you should prefer batch sizes that divide dataset size or use `DropLast`.

To allow better work with batches, you should use `BatchPlan`,
which records the batch sizes and lazy loading actions for one dataset pass.
`prepareBatchStep` uses those sizes to prepare a function for each specialization.
Bind the resulting executor outside the batch and epoch loops, then reuse it:

```haskell
plan = NN.planBatches 32 NN.KeepLast dataset
step = NN.prepareBatchStep backend plan trainingStep

forM_ (NN.batchLoads plan) $ \load -> do
    someBatch <- load
    NN.withBatch someBatch $ \batch ->
        case NN.runBatch step state batch of
            Left unexpected -> print unexpected
            Right result -> useResult result
```

Reusing the executor retains those prepared computations and lets backend reuse cached kernels across shapes.
If a batch has a size not listed in the plan,
`runBatch` returns `UnexpectedSize` (you could specialize or fail in that case).

#### Training

`Synapse` provides regression losses such as MSE, MAE, Huber, and Smooth L1.
Classification losses include cross entropy, binary cross entropy, and negative log likelihood.
Each loss accepts a reduction such as `T.mean` or `T.sum`, so you choose how per-example errors are combined.
Metrics include regression measures and classification accuracy, F1, confusion matrices and many more.

Custom losses and metrics are ordinary functions over tensors. For example,
a weighted squared error can be written directly from tensor operations:

```haskell
weightedMSE weights predictions targets =
    let errors = predictions - targets
     in T.mean (weights * errors * errors)
```

A training step reads the current parameters and optimizer state, computes the
loss and gradients, updates both, and stores them for the next batch:

```haskell
trainingStep features targets = do
    (parameters, optimizerState) <- get
    let objective p =
            NN.mse T.mean
                (predict p (AG.constant features))
                (AG.constant targets)
        (loss, gradients) = AG.backprop objective parameters
        (updated, nextOptimizerState) =
            NN.runState (NN.optimizerStep optimizer parameters gradients) optimizerState
    put (updated, nextOptimizerState)
    pure loss
```

The training loop runs this step through the prepared batch executor.
For validation, use the same forward computation to collect a loss and metrics without updating parameters:

```haskell
validationStep parameters features targets =
    let logits = predict parameters features
     in ( NN.unsafeCrossEntropy @1 T.mean logits targets
        , NN.accuracy @1 logits targets
        , NN.f1 @1 NN.Macro logits targets
        )
```

#### Weights and Safetensors

`Weights` is a flat map from tensor field names to tensor values.
It is a portable view of a model's parameters,
separate from the record and layer types used by the model itself.
`extractWeights` converts a parameter structure into this map;
`loadWeights` fills a parameter template from it.
The template supplies the expected structure, shapes, and element types.

Once parameters have been extracted, they can be written to a Safetensors file:

```haskell
weights = NN.extractWeights parameters
writeSafetensors "model.safetensors" weights
```

For evaluation, read the file and load the tensors into an initialized parameter template:

```haskell
weights <- either (fail . show) pure =<< readSafetensors "model.safetensors"
parameters <-
    either (fail . show) pure $
        NN.loadWeights @(Parameters T.Tensor) weights (T.run backend emptyParameters)
```

Pass the loaded parameters to the model's forward function for inference.

#### Examples

The examples cover the main parts of `Synapse`'s neural network API:

* [Regression][regression-example] trains a dense network on a nonlinear function.
* [Classification][classification-example] trains a classifier and reports common metrics.
* [Convolution][convolution-example] builds a residual convolutional network with BatchNorm.
* [Recurrent][recurrent-example] carries state through an LSTM sequence.
* [Transformer][transformer-example] combines embeddings, attention, residuals, and a feed-forward block.

## 📊 Benchmarks 📊

The benchmark runner compares `Synapse` with `Hasktorch` and `Grenade` on
classification tasks with several model architectures:

* [Iris][iris-benchmark] uses an MLP.
* [MNIST MLP][mnist-mlp-benchmark] uses fully connected layers.
* [MNIST CNN][mnist-cnn-benchmark] uses a convolutional network.
* [MNIST LSTM][mnist-lstm-benchmark] processes each image as a sequence.
* [MNIST ViT][mnist-vit-benchmark] uses a vision transformer.

Each task runs for a fixed number of epochs.
The default is 200 epochs for Iris and 3 for MNIST.
The runner repeats each framework five times by default and rotates their execution order between trials.
Reports summarize timing with the median, minimum, and maximum across trials.

The runner records the first training step separately from total training time.
The first-step measurement includes initial execution and any lazy compilation triggered by that step.
Reports therefore include total training time and training time after subtracting the first step.
Each per-trial JSON also records setup, first-step, training, and whole-process time.

Iris and MNIST MLP/CNN run in `Synapse`, `Hasktorch`, and `Grenade`.
`Grenade`'s LSTM differs in implementation from what `Synapse` `Hasktorch` use,
so we ...
`Grenade` does not have transformers, so ViT task runs in `Synapse` and `Hasktorch` only.

The runner checks numerical outputs against `Hasktorch` for each trial for a reference.

### CPU benchmarks

### GPU benchmarks

## 📖 Future plans 📖

`Synapse` already covers many of the ideas that used to be on this list.
Possible next steps include:

* **Optimized numerical backends**

    Explore optional oneDNN CPU and cuDNN GPU integrations for operations such as convolutions.
    The goal would be to use tuned kernels where they help,
    while keeping the existing backend available as a portable fallback.

* **More performance!**

    There is no such thing as too much performance;
    although `Synapse` is pretty optimized, there is always possibility for more fine-tuning.

* **Numeric formats**

    `Synapse` does not support half-precision floating-point formats, such as fp16 and bf16,
    and does not support quantization currently.
    This could be a prominent upgrade.

* **Training workflow utilities**

    There is still lots of QoL things that `Synapse` misses on.
    Small additions for common training needs such as learning-rate schedules,
    checkpointing, validation workflows, etc. would definitely make experience better.

* **Interoperability with other frameworks**

    Improve ways to exchange model weights and datasets with other tools,
    so users can bring existing data and use models trained outside `Synapse` more easily,
    without need for manual conversion.

## ❤️ Contributions ❤️

`Synapse` library would benefit from every contribution to it: docs, code, small advertisement - you name it.

If you want to help me in the development, you could always contact me - 
my [GitHub profile](https://github.com/JktuJQ) has links to my socials.

[neural]: https://hackage.haskell.org/package/neural
[grenade]: https://hackage.haskell.org/package/grenade
[hasktorch]: https://hackage.haskell.org/package/hasktorch

[jax-custom-vjp]: https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html

[accelerate]: https://hackage.haskell.org/package/accelerate
[backprop]: https://hackage.haskell.org/package/backprop

[tensor-example]: examples/overview/Tensor.lhs
[autograd-example]: examples/overview/Autograd.lhs
[regression-example]: examples/overview/NN/Regression.lhs
[classification-example]: examples/overview/NN/Classification.lhs
[convolution-example]: examples/overview/NN/Convolution.lhs
[recurrent-example]: examples/overview/NN/Recurrent.lhs
[transformer-example]: examples/overview/NN/Transformer.lhs

[iris-benchmark]: benchmarks/frameworks/synapse/Iris.hs
[mnist-mlp-benchmark]: benchmarks/frameworks/synapse/MnistMlp.hs
[mnist-cnn-benchmark]: benchmarks/frameworks/synapse/MnistCnn.hs
[mnist-lstm-benchmark]: benchmarks/frameworks/synapse/MnistLstm.hs
[mnist-vit-benchmark]: benchmarks/frameworks/synapse/MnistVit.hs
