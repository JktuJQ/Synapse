Nearest centers
===============

Given points and centers on a plane, find the nearest center to each point.
The input tensors have shapes [n, 2] and [m, 2], where n is the number of points
and m is the number of centers.

> {-# LANGUAGE DataKinds #-}
> {-# LANGUAGE FlexibleContexts #-}
> {-# LANGUAGE ScopedTypeVariables #-}
> {-# LANGUAGE TypeApplications #-}
>
> module Main (main) where
>
> import System.Environment (getArgs)
> import GHC.TypeNats (KnownNat)
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
>         _ -> ioError $ userError "Usage: tensor-example [--cpu | --gpu]"

The result contains an [n, m] matrix of squared distances and one nearest-center index per point.
Shapes are part of tensor types, so n and m are known when the function is compiled.
We require a nonempty center axis for argmin.
Each prepared computation accepts inputs of its chosen shapes;
other values of n or m require a separately prepared computation,
although internal cache can reuse compiled kernels across different shapes.

> nearestCenters
>     :: forall n m
>      . ( KnownNat n, KnownNat m
>        , T.CanReduceNonEmpty '[1] '[n, m]
>        )
>     => T.Tensor '[n, 2] Double
>     -> T.Tensor '[m, 2] Double
>     -> (T.Tensor '[n, m] Double, T.Tensor '[n] Int)
> nearestCenters points centers =

Insert a center axis into the points and a point axis into the centers.
The new shapes [n, 1, 2] and [1, m, 2] identify which axis each input varies along.

>     let pointRows = T.unsqueeze @1 points
>         centerColumns = T.unsqueeze @0 centers

Broadcast both tensors to [n, m, 2].
Each pair of coordinates can now be subtracted to obtain the displacement from a center to a point.

>         differences = T.broadcast @'[n, m, 2] pointRows
>                     - T.broadcast @'[n, m, 2] centerColumns

Sum the squared coordinates along the last axis.
Each row of the [n, m] result contains the squared distances from one point to all centers.

>         distances = T.sumAxes @'[2] (differences * differences)

Choose the smallest distance along the center axis.
The result has one index per point.

>         nearest = T.argmin @1 distances
>     in (distances, nearest)

For this run, choose four points and three centers.
Create the input data, prepare the computation for the selected backend, and execute it.
TensorData is materialized; its elements can be printed with toList.
The distance matrix is displayed in row-major order.

> main :: IO ()
> main = do
>     backend <- chooseBackend
>     let points = T.fromListData @'[4, 2] $ concat
>             [[0, 1], [3, 1], [4, 5], [-1, -1 :: Double]]
>         centers = T.fromListData @'[3, 2] $ concat
>             [[0, 0], [4, 0], [4, 4 :: Double]]
>         execute = T.run2 backend nearestCenters
>         (distances, nearest) = execute points centers
>
>     putStrLn $ "Backend: " ++ show backend
>     putStrLn "Squared distances (one row per point):"
>     print $ T.toList distances
>     putStrLn "Nearest center indices:"
>     print $ T.toList nearest
