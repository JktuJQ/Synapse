# ![Synapse logo](SynapseLogo.png) Synapse ![Synapse logo](SynapseLogo.png)

`Synapse` is a machine learning library written in pure Haskell,
that makes creating and training neural networks an easy job.

## 🔨 Design 🔨

Goals of `Synapse` library are to provide interface that is:

* easily extensible
* simple
* powerful

Haskell ecosystem only offers a few of machine learning libraries,
but all of them have very complicated interface -
[neural](https://hackage.haskell.org/package/neural) does not have too much extensibility and its typing is pretty hard,
[grenade](https://hackage.haskell.org/package/grenade), although very powerful, uses a lot of type trickery that is impossible to reason about for beginners
and [hasktorch](https://hackage.haskell.org/package/hasktorch) is a wrapper that is not as convenient as one might want.

`Synapse` tries to resemble Python's [Keras](https://keras.io/api/) API,
which is a unified interface for backends such as [Pytorch](https://pytorch.org/) and [Tensorflow](https://www.tensorflow.org/).

Even though Python code practices usually are not what we want to see in Haskell code,
mantaining the same level of accessibility and flexibility is what `Synapse` is focused on.

## 💻 Usage 💻

`Synapse` comes batteries-included with its own matrices,
autodifferentiation system,
and neural networks building blocks.

### `Vec`s and `Mat`s

Clever usage of Haskell's typeclasses allows operations to be written 'as is' for different types.

Following example emulates dense layer forward propagation on plain matrices.

```Haskell
input   = M.rowVec $ V.fromList [1.0, 2.0, 3.0]

weights = M.replicate (3, 3) 1.0
bias    = M.rowVec $ V.replicate 3 0.0

output  = tanh (input `matMul` weights `addMatRows` bias)
```

### Symbolic operations and autograd system

`Synapse` autograd system implements reverse-mode dynamic automatic differentiation, where graph of operations is created on the fly.
Its usage is much simpler than [ad](https://hackage.haskell.org/package/ad),
and it is easily extensible - all you need is to define a function that produces `Symbol` which will hold gradients of
that function and that is it!
There is a [backprop](https://hackage.haskell.org/package/backprop) library that has a very similar implementation to `Synapse`'s one,
so if you are familiar with [backprop](https://hackage.haskell.org/package/backprop) you will have no problems using `Synapse.Autograd` module.

Speaking of the previous example - you might want to record gradients of those operations, and that is just as easy:

```Haskell
input   = symbol (SymbolIdentifier "input") $ M.rowVec $ V.fromList [1.0, 2.0, 3.0]

weights = symbol (SymbolIdentifier "weights") $ M.replicate (3, 3) 1.0
bias    = symbol (SymbolIdentifier "bias") $ M.rowVec $ V.replicate 3 0.0

output  = tanh (input `matMul` weights `addMatRows` bias)
```

You just need to set the names for your symbolic variables, and you are good to go - `Synapse` will take care of the rest.

Look at the `Synapse` implementation of some common operations:

```Haskell
(+) a b = symbolicBinaryOp (+) a b [(a, id), (b, id)]
(*) a b = symbolicBinaryOp (*) a b [(a, (* b)), (b, (a *))]

exp x = symbolicUnaryOp exp x [(x, (* exp x))]
sin x = symbolicUnaryOp sin x [(x, (* cos x))]
```

Provided functions (`symbolicUnaryOp`, `symbolicBinaryOp`) expect an operation that will be performed on *values* of those symbols,
symbols themselves and a list of tuples,
where each tuple represents a gradient (the symbol which gradient is taken and function that implements chain rule - multiplies already calculated gradient (*symbol*) by the gradient of the function (another *symbol*)).
Using those, defining new symbolic operations is very easy,
and you should note that any composition of symbolic operations is itself a symbolic operation.

This implementation is really easy to use:

```Haskell
a = symbol (SymbolIdentifier "a") 4
b = symbol (SymbolIdentifier "b") 3
c = renameSymbol (SymbolIdentifier "c") ((a * a) + (b * b))

getGradientsOf (a * (b + b) * a) `wrt` a  -- == 4 * 4 * 3
getGradientsOf (c * c) `wrt` c            -- == 2 * 25

nthGradient 2 (a * (b + b) * a) a         -- == 4 * 3
nthGradient 4 (sin c) c                   -- == sin c
```

`Synapse` does not care what is the type of your symbols: it might be `Int`, `Double`, `Vec`, `Mat`, anything that instantiates `Symbolic` typeclass -
types just need to match with each other and with types of operations.

### Neural networks

`Synapse` takes as much of [Keras](https://keras.io/api/) API as it is possible, but is also provides additional abstractions leveraging Haskell's type system.

#### Functions

Everything that is a function in a common sense, is a function too in `Synapse`.

`Activation`, `Loss`, `Metric`, `LearningRate`, `Constraint`, `Initializer`, `Regularizer` newtypes
just wrap any plain Haskell function with needed type!

That means that to create new activation function, loss function, etc. you just need to create Haskell function with appropriate constrains.

```Haskell
sigmoid :: (Symbolic a, Floating a) => a -> a
sigmoid x = recip $ exp (negate x) + symbolicOne x

sigmoidActivation :: (Symbolic a, Floating a) => Activation a
sigmoidActivation = Activation sigmoid
```

`symbolicOne` function represents constant that corresponds to identity element for given `x`.
Usage of const literal `1.0` does not work, because that identity element also needs to know `x`s dimensions.
If `x` is matrix, you need to get `M.replicate (M.size x) 1.0`,
not the singleton `1.0`.
Writing additional constraints like `Symbolic` and having to create constants using `symbolicOne` might seem tedious,
but that ensures type safety.

You can also specialize the function:

```Haskell
type ActivationFn a = SymbolMat a -> SymbolMat a

sigmoid :: (Symbolic a, Floating a) => ActivationFn a
sigmoid x = recip $ exp (negate x) +. 1.0

sigmoidActivation :: (Symbolic a, Floating a) => Activation a
sigmoidActivation = Activation sigmoid
```

Even with all those limitations, it is still easy to create your own functions for any task.

#### Layer system

`AbstractLayer` typeclass is the most low-leveled abstraction of entire `Synapse` library.
3 functions (`getParameters`, `updateParameters` and `symbolicForward`) are the backbone of entire neural networks interface.
Docs on that typeclass as well as docs on those functions extensively describe invariants that `Synapse` expects from their implementations.

With the help of `Layer` existential datatype `Synapse` is able to
build networks from any types that are instances of `AbstractLayer` typeclass,
which means that this system is easily extendable.

`Dense` layer, for example, supports regularizers, constraints, recording gradients on its forward operations, and that is built upon those 3 functions.

#### Training

Here is the `train` function signature:

```Haskell
train
    :: (Symbolic a, Floating a, Ord a, Show a, RandomGen g, AbstractLayer model, Optimizer optimizer)
    => model a
    -> optimizer a
    -> Hyperparameters a
    -> Callbacks model optimizer a
    -> g
    -> IO (model a, [OptimizerParameters optimizer a], Vec (RecordedMetric a), g)
```

Let's break it down into pieces and examine that function.

#### Models

`model` represents any `AbstractLayer` instance, so it is the model with parameters that are going to be trained.

Any layer can be a model, but more commonly you would use `SequentialModel`.
`SequentialModel` is a newtype that wraps list of `Layer`s.
`buildSequentialModel` function builds the model, ensuring that dimensions of layers match.

That is achieved by `LayerConfiguration` type alias and corresponding functions like `layerDense` and `layerActivation`.
`LayerConfiguration` represents functions that are able to build a new layer upon the other layer, using information about its output dimension.

```Haskell
layers = [ Layer . layerDense 1
         , Layer . layerActivation (Activation tanh)
         , Layer . layerDense 1
         ] :: [LayerConfiguration (Layer Double)]
```

You just write your layers like that and let `buildSequentialModel` to figure out how to compose them.

It would look like this:

```Haskell
model = buildSequentialModel
            (InputSize 1)
            [ Layer . layerDense 1
            , Layer . layerActivation (Activation tanh)
            , Layer . layerDense 1
            ] :: SequentialModel Double
```

`InputSize` indicates size of input that will be supported by this model.
Model can take any matrix of size `(n, i)`, where `i` was supplied as `InputSize i` when the model was built.

Since `AbstractLayer` instance is a trainable model and `SequentialModel` is a model, it means that it is also an instance `AbstractLayer`.
Some layers are inherently a composition of other layers (LSTM layers are the example) and `Synapse` supports this automatically.

#### Optimizers

`optimizer` represents any `Optimizer` instance.
Any optimizer has its parameters (`OptimizerParameters`) which it uses to update parameters of a model.

Update is done with the functions `optimizerUpdateStep` and `optimizerUpdateParameters`.
The second function is a mass update, so it needs gradients on all parameters of the model which are represented by symbolic matrices, while the first updates only one parameters which does not need to be symbolic, due to supplied exact gradient value.

It is pretty easy to implement your own optimizer.
See how `Synapse` implements `SGD`:

```Haskell
data SGD a = SGD
    { sgdMomentum :: a
    , sgdNesterov :: Bool
    } deriving (Eq, Show)

instance Optimizer SGD where
    type OptimizerParameters SGD a = Mat a

    optimizerInitialParameters _ parameter = zeroes (M.size parameter)

    optimizerUpdateStep (SGD momentum nesterov) (lr, gradient) (parameter, velocity) = (parameter', velocity')
      where
        velocity' = velocity *. momentum - gradient *. lr

        parameter' = if nesterov
                     then parameter + velocity' *. momentum - gradient *. lr
                     else parameter + velocity'
```

#### Hyperparameters

Any training has some hyperparameters that configure that training.

```Haskell
data Hyperparameters a = Hyperparameters
    { hyperparametersEpochs       :: Int
    , hyperparametersBatchSize    :: Int

    , hyperparametersDataset      :: VecDataset a

    , hyperparametersLearningRate :: LearningRate a
    , hyperparametersLoss         :: Loss a

    , hyperparametersMetrics      :: Vec (Metric a)
    }
```

Those hyperparameters include the number of epochs, batch size,
dataset of vector samples (vector input and vector output),
learning rate function, loss function
and metrics that will be recorded during training.

#### Callbacks

`Synapse` allows 'peeking' in the training process using callbacks system.

Several type aliases
(`CallbackFnOnTrainBegin`,
`CallbackFnOnEpochBegin`,
`CallbackFnOnBatchBegin`,
`CallbackFnOnBatchEnd`,
`CallbackFnOnEpochEnd`,
`CallbackFnOnTrainEnd`)
represent functions that take mutable references to training parameters and do something with them (read, print/save, modify, etc.).

Callback system interface should be used with caution, because some changes might break the training completely,
but nonetheless, it is a very powerful instrument.

#### Training process

Training itself consists of following steps:

1. Training beginning (setting up initial parameters of model and optimizer)
2. Epoch training (shuffling, batching and processing of the dataset)
3. Batch training (update of parameters based on the result of batch processing, recording of metrics)
4. Training end (collecting results of training)

All of that is handled by the `train` function.

Here is an example of sine wave approximator which you could find at [tests](./test/) directory:

```Haskell
let sinFn x = (-3.0) * sin (x + 5.0)
let model = buildSequentialModel (InputSize 1) [ Layer . layerDense 1
                                               , Layer . layerActivation (Activation cos)
                                               , Layer . layerDense 1
                                               ] :: SequentialModel Double
let dataset = Dataset $ V.fromList $ [Sample (singleton x) (sinFn $ singleton x) | x <- [-pi, -pi+0.2 .. pi]]
(trainedModel, _, losses, _) <- train model
                                   (SGD 0.2 False)
                                   (Hyperparameters 500 16 dataset (LearningRate $ const 0.01) (Loss mse) V.empty)
                                   emptyCallbacks
                                   (mkStdGen 1)
_ <- plot (PNG "test/plots/sin.png")
          [ Data2D [Title "predicted sin", Style Lines, Color Red] [Range (-pi) pi] [(x, unSingleton $ forward (singleton x) trainedModel) | x <- [-pi, -pi+0.05 .. pi]]
          , Data2D [Title "true sin", Style Lines, Color Green] [Range (-pi) pi] [(x, sinFn x) | x <- [-pi, -pi+0.05 .. pi]]
          ]
let unpackedLosses = unRecordedMetric (unsafeIndex losses 0)
let lastLoss = unsafeIndex unpackedLosses (V.size unpackedLosses - 1)
assertBool "trained well enough" (lastLoss < 0.01)
```

##### Prefix system

`Synapse` manages gradients and parameters for layers with erased type information using prefix system.

`SymbolIdentifier` is a prefix for name of symbolic parameters that are used in calculation.
Every used parameter should have unique name to be recognised by the autograd -
it must start with given prefix and end with the numerical index of said parameter.
For example 3rd layer with 2 parameters (weights and bias) should
name its weights symbol `"ml3w1"` and name its bias symbol `"ml3w2"` (`"ml3w"` prefix will be supplied).

Prefix system along with layer system require to carefully ensure all the invariants that `Synapse` imposes if you are willing to extend them (write your own layers, training loops, etc.).
But the user of this library should not worry about those getting in the way, because they are hidden behind an abstraction.

## 📖 Future plans 📖

`Synapse` library is still under development and there is work to be done on:

* **Performance**

    `Synapse` 'brings its own guns' and, although it makes the library independent,
it could make `Synapse` to miss on some things that are battle-tested and tuned to performance.

    That is especially true for `Synapse.Tensors` implementations of `Vec` and `Mat`.
Those are built upon [vector](https://hackage.haskell.org/package/vector) library,
which is good, but it is not suitable for heavy numerical calculations.

    [hmatrix](https://hackage.haskell.org/package/hmatrix) which offers numerical computations based on BLAS and LAPACK is way more efficient.
It would be great if `Synapse` library could work with any matrix backend.

* **Tensors**

    It is really a shame that `Synapse.Tensors` does not have actual tensors.
`Tensor` datatype would allow to get rid of `Vec` and `Mat` datatypes in favour of more powerful abstraction.
Tensor broadcasting could also be created to address issues that `Symbolic` typeclass is trying to solve.
`Tensor` datatype could even present a unified frontend for any matrix backend that `Synapse` could use.

* **GPU support**

    This clause addresses all of the issues above.
It would severely increase performance of `Synapse` library,
and it would greatly work with backend-independent tensors.
Haskell ecosystem offers great [accelerate](https://hackage.haskell.org/package/accelerate) library which could help with all those problems.

* **More out-of-the-box solutions**

    At this point, `Synapse` does not offer a wide variety of layers, activations, models, optimizers out-of-the-box.
Suppling more of them would definitely help.

* **Developer abstractions**

    Currently, `Synapse`'s backbones are `SymbolIdentifier`s and implementations of `AbstractLayer`,
which might be cumbersome for developers and be a bit unreliable.
If those systems could work naturally, without some hardcoding of values, it would be great.

* **More monads!**

    `Synapse` library does not use a lot of Haskell instruments, like optics, monad transformers, etc.
Although it makes the library easy for beginners,
I am sure that some of those instruments can offer richer expressiveness for the code
and also address the issue of 'Developer abstractions'.

## ❤️ Contributions ❤️

`Synapse` library would benefit from every contribution to it: docs, code, small advertisement - you name it.

If you want to help me in the development, you could always contact me -
my [GitHub profile](https://github.com/JktuJQ) has links to my socials.
