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
        ["train", dataset, weights] -> train backend dataset weights 200
        ["train", dataset, weights, count]
            | Just epochs <- readMaybe count ->
                train backend dataset weights epochs
        ["evaluate", dataset, weights] -> evaluateModel backend dataset weights
        _ ->
            fail $
                unlines
                    [ "Usage: iris-example [--cpu | --gpu] train IRIS_DATA WEIGHTS [EPOCHS]"
                    , "       iris-example [--cpu | --gpu] evaluate IRIS_DATA WEIGHTS"
                    ]
