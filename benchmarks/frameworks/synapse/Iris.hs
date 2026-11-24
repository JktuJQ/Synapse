{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Main (main) where

import Control.Exception (evaluate)
import Control.Monad (foldM, unless)
import GHC.Clock (getMonotonicTimeNSec)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat)
import System.Environment (getArgs)
import System.FilePath ((</>))
import Text.Printf (printf)
import Text.Read (readMaybe)

import Synapse.Autograd qualified as AG
import Synapse.NN qualified as NN
import Synapse.Tensor qualified as T

data Parameters t = Parameters
    { hidden :: NN.Dense 4 16 t Double
    , output :: NN.Dense 16 3 t Double
    }
    deriving (Generic)

instance T.TensorStructure (Parameters T.Tensor) where
    type WithTensor t (Parameters T.Tensor) = Parameters t

predict
    :: (KnownNat n, T.RealFloatTensorOps t Double)
    => Parameters t
    -> t '[n, 4] Double
    -> t '[n, 3] Double
predict parameters = NN.dense (output parameters) . NN.tanh . NN.dense (hidden parameters)

trainingStep
    :: Parameters T.Tensor
    -> T.Tensor '[120, 4] Double
    -> T.Tensor '[120] Int
    -> (T.Tensor '[] Double, Parameters T.Tensor)
trainingStep parameters features targets =
    let objective
            :: AG.Context s
            => Parameters (AG.Tensor s)
            -> AG.Tensor s '[] Double
        objective p = NN.unsafeCrossEntropy @1 T.mean (predict p $ AG.constant features) targets
        (loss, gradients) = AG.backprop objective parameters
        updated = NN.evalState (NN.optimizerStep (NN.SGD 0.1) parameters gradients) ()
     in (loss, updated)

readSamples
    :: FilePath
    -> Int
    -> IO ([Double], [Int])
readSamples path expected = do
    rows <- lines <$> readFile path
    samples <- mapM parseRow rows
    unless (length samples == expected) $ fail $ path ++ ": unexpected sample count"
    pure (concatMap fst samples, map snd samples)
  where
    parseRow row = case words $ map (\c -> if c == ',' then ' ' else c) row of
        [a, b, c, d, label] -> case (traverse readMaybe [a, b, c, d], readMaybe label) of
            (Just features, Just category) | category >= 0 && category < 3 -> pure (features, category)
            _ -> fail $ path ++ ": invalid sample"
        _ -> fail $ path ++ ": expected four features and a label"

readParameters
    :: FilePath
    -> IO (Parameters T.TensorData)
readParameters path = do
    tokens <- words <$> readFile path
    values <- maybe (fail "Invalid initial weights") pure $ traverse readMaybe tokens
    unless (length values == 131) $ fail "Expected 131 initial parameters"
    let (first, rest) = splitAt 64 values
        (firstBias, remaining) = splitAt 16 rest
        (second, secondBias) = splitAt 48 remaining
    pure $
        Parameters
            (NN.Dense (T.fromListData first) $ T.fromListData firstBias)
            (NN.Dense (T.fromListData second) $ T.fromListData secondBias)

parameterValues
    :: Parameters T.TensorData
    -> [Double]
parameterValues (Parameters (NN.Dense first firstBias) (NN.Dense second secondBias)) =
    T.toList first ++ T.toList firstBias ++ T.toList second ++ T.toList secondBias

seconds
    :: IO Double
seconds = (/ 1e9) . fromIntegral <$> getMonotonicTimeNSec

main :: IO ()
main = do
    args <- getArgs
    case args of
        [directory, count] | Just epochs <- readMaybe count -> runIris directory epochs
        _ -> fail "Usage: synapse-iris DATA_DIRECTORY EPOCHS"

runIris :: FilePath -> Int -> IO ()
runIris directory epochs = do
    unless (epochs > 0) $ fail "Epoch count must be positive"
    (trainFeatures, trainTargets) <- readSamples (directory </> "train.csv") 120
    (testFeatures, testTargets) <- readSamples (directory </> "test.csv") 30
    parameters <- readParameters $ directory </> "weights.txt"
    let features = T.fromListData @'[120, 4] trainFeatures
        targets = T.fromListData @'[120] trainTargets
        testInput = T.fromListData @'[30, 4] testFeatures
        testLabels = T.fromListData @'[30] testTargets
        infer = T.run T.defaultCPU $ \p x -> predict p x
        step = T.run T.defaultCPU trainingStep
        validate = T.run T.defaultCPU $ \p x y ->
            let logits = predict p x
             in (NN.unsafeCrossEntropy @1 T.mean logits y, NN.accuracy @1 logits y)
    _ <- evaluate $ sum trainFeatures + sum testFeatures + sum (parameterValues parameters)
    setupStart <- seconds
    let logits = T.toList $ infer parameters features
    _ <- evaluate $ sum logits
    setupEnd <- seconds
    (firstLoss, firstParameters) <- forceStep step parameters features targets
    firstEnd <- seconds
    (finalParameters, losses) <-
        foldM
            ( \(p, history) _ -> do
                (loss, next) <- forceStep step p features targets
                pure (next, loss : history)
            )
            (firstParameters, [firstLoss])
            [2 .. epochs]
    trainingEnd <- seconds
    let (testLoss, accuracy) = validate finalParameters testInput testLabels
    measuredLoss <- evaluate $ sum $ T.toList testLoss
    measuredAccuracy <- evaluate $ sum $ T.toList accuracy
    printf
        ( "{\"framework\":\"synapse\",\"device\":\"%s\",\"dtype\":\"float64\",\"epochs\":%d,"
            ++ "\"setup_seconds\":%.9f,\"first_step_seconds\":%.9f,\"training_seconds\":%.9f,"
            ++ "\"initial_logits\":%s,\"first_step_parameters\":%s,\"losses\":%s,"
            ++ "\"test_loss\":%.12f,\"test_accuracy\":%.12f}\n"
        )
        ("cpu" :: String)
        epochs
        (setupEnd - setupStart)
        (firstEnd - setupEnd)
        (trainingEnd - setupEnd)
        (show logits)
        (show $ parameterValues firstParameters)
        (show $ reverse losses)
        measuredLoss
        measuredAccuracy
  where
    forceStep step parameters features targets = do
        let (loss, updated) = step parameters features targets
        value <- evaluate $ sum $ T.toList loss
        _ <- evaluate $ sum $ parameterValues updated
        pure (value, updated)
