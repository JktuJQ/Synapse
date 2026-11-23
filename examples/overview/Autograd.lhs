Fitting a quadratic curve
=========================

Given five observations from y = x^2 + 2*x + 1,
fit the coefficients of a quadratic curve y = a*x^2 + b*x + c.
We start at zero and use gradient descent to minimize the mean squared error between predictions and observations.

> module Main (main) where
>
> import Control.Monad (foldM, when)
> import GHC.TypeNats (KnownNat)
> import System.Environment (getArgs)
> import qualified Synapse.Autograd as AG
> import qualified Synapse.Tensor as T

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
>         _ -> ioError $ userError "Usage: autograd-example [--cpu | --gpu]"

Each coefficient is a scalar tensor.
During differentiation, parameters use AG.Tensor,
while the observations stay ordinary tensors: they are not variables we want to optimize.
The number of observations n is part of the input shapes.

> type Coefficients a = (a, a, a)

> loss
>     :: (AG.Context s, KnownNat n)
>     => T.Tensor '[n] Double
>     -> T.Tensor '[n] Double
>     -> Coefficients (AG.Tensor s '[] Double)
>     -> AG.Tensor s '[] Double
> loss inputs targets (a, b, c) =

Lift the observations into the differentiated computation as constants.
The scalar coefficients are broadcast to match the inputs,
then we evaluate the quadratic curve for every observation.

>     let x = AG.constant inputs
>         y = AG.constant targets
>         predicted = T.broadcast a * x * x + T.broadcast b * x + T.broadcast c

Subtract the observed values and take the mean of the squared errors.
The result is a scalar tensor; backpropagation computes its gradient with respect to all three parameters.
As with T.mean, an empty set of observations produces NaN.

>         residual = predicted - y
>     in T.mean (residual * residual)

One training step calculates both the loss and its gradient,
then moves each coefficient in the opposite direction.
The observations are supplied separately so the same step can work with other values of the same shape.

> trainingStep
>     :: KnownNat n
>     => T.Tensor '[n] Double
>     -> T.Tensor '[n] Double
>     -> Coefficients (T.Tensor '[] Double)
>     -> ( T.Tensor '[] Double
>        , Coefficients (T.Tensor '[] Double)
>        , Coefficients (T.Tensor '[] Double)
>        )
> trainingStep inputs targets parameters =
>     let (value, (da, db, dc)) = AG.backprop (loss inputs targets) parameters
>         (a, b, c) = parameters
>         learningRate = 0.05
>         next = (a - learningRate * da, b - learningRate * db, c - learningRate * dc)
>     in (value, (da, db, dc), next)

Prepare the training step once with run3.
Each execution accepts materialized inputs, targets, and parameters, and returns materialized results.
This run uses five observations; another input size requires a separately prepared step,
although internal cache can reuse compiled kernels across different shapes.
First inspect the initial gradient, then use the same prepared step throughout the loop.

> main :: IO ()
> main = do
>     backend <- chooseBackend
>     let inputs = T.fromListData @'[5] [-2, -1, 0, 1, 2 :: Double]
>         targets = T.fromListData @'[5] [1, 0, 1, 4, 9 :: Double]
>         zero = T.fromListData @'[] [0 :: Double]
>         initial = (zero, zero, zero)
>         step = T.run3 backend trainingStep
>         (initialLoss, (da, db, dc), _) = step inputs targets initial
>
>     putStrLn $ "Backend: " ++ show backend
>     putStrLn $ "Initial loss: " ++ show (T.toList initialLoss)
>     putStrLn $ "Initial gradient (a, b, c): " ++ show (map T.toList [da, db, dc])
>
>     (a, b, c) <- foldM (\parameters iteration -> do
>         let (value, _, next) = step inputs targets parameters
>         when (iteration `mod` 50 == 0) $
>             putStrLn $ "Step " ++ show iteration ++ ": loss = " ++ show (T.toList value)
>         pure next) initial [1 .. 250 :: Int]
>
>     putStrLn $ "Fitted parameters (a, b, c): " ++ show (map T.toList [a, b, c])
