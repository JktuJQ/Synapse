module Main (main) where

import System.Environment (getArgs)
import Text.Read (readMaybe)

import Evaluate (evaluateModel)
import Synapse.Tensor qualified as T
import Train (train)

main :: IO ()
main = do
    arguments <- getArgs
    let (backend, args) = case arguments of
            "--cpu" : rest -> (T.defaultCPU, rest)
            "--gpu" : rest -> (T.defaultGPU, rest)
            _ -> (T.defaultCPU, arguments)
    case args of
        ["train", dataset, weights] -> train backend dataset weights 3
        ["train", dataset, weights, count]
            | Just epochs <- readMaybe count ->
                train backend dataset weights epochs
        ["evaluate", dataset, weights] -> evaluateModel backend dataset weights
        _ ->
            fail $
                unlines
                    [ "Usage: mnist-example [--cpu | --gpu] train DATA_DIRECTORY WEIGHTS [EPOCHS]"
                    , "       mnist-example [--cpu | --gpu] evaluate DATA_DIRECTORY WEIGHTS"
                    ]
