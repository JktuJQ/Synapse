module Task (Task (..), Model (..), modelName, taskName, parameterCount, initialWeights) where

import Data.List (mapAccumL)
import System.Random (mkStdGen, uniformR)

data Model = MLP | CNN | LSTM | ViT
    deriving (Bounded, Enum, Eq)

modelName :: Model -> String
modelName MLP = "mlp"
modelName CNN = "cnn"
modelName LSTM = "lstm"
modelName ViT = "vit"

data Task = Iris | Mnist Model

taskName :: Task -> String
taskName Iris = "iris-mlp"
taskName (Mnist model) = "mnist-" ++ modelName model

data Weights = Xavier Int Int Int | Constant Int Double

dense :: Int -> Int -> [Weights]
dense inputs outputs = [Xavier inputs outputs (inputs * outputs), Constant outputs 0]

weightLayout :: Task -> [Weights]
weightLayout Iris = dense 4 16 ++ dense 16 3
weightLayout (Mnist MLP) = dense 784 64 ++ dense 64 10
weightLayout (Mnist CNN) = [Xavier 25 100 100, Constant 4 0] ++ dense 576 10
weightLayout (Mnist LSTM) =
    [Xavier 28 64 (28 * 64), Xavier 16 64 (16 * 64), Constant 64 0, Constant 64 0] ++ dense 16 10
weightLayout (Mnist ViT) =
    concatMap (uncurry dense) [(16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 32), (32, 16), (16, 10)]
        ++ [Constant 16 1, Constant 16 0, Constant 16 1, Constant 16 0]

parameterCount :: Task -> Int
parameterCount = sum . map size . weightLayout
  where
    size (Xavier _ _ n) = n
    size (Constant n _) = n

initialWeights :: Int -> Task -> [Double]
initialWeights seed = concat . snd . mapAccumL initialize (mkStdGen seed) . weightLayout
  where
    initialize generator (Constant count value) = (generator, replicate count value)
    initialize generator (Xavier inputs outputs count) =
        let bound = sqrt (6 / fromIntegral (inputs + outputs))
            sample g () = let (value, next) = uniformR (-bound, bound) g in (next, value)
         in mapAccumL sample generator (replicate count ())
