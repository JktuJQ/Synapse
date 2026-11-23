{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

module Main (main) where

import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Monad (foldM, unless, when)
import Data.Version (showVersion)
import GHC.Clock (getMonotonicTimeNSec)
import System.Environment (getArgs)
import System.Info (compilerName, compilerVersion)
import Text.JSON qualified as JSON
import Data.Maybe (fromMaybe)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import System.FilePath ((</>))
import Text.Read (readMaybe)

import Grenade

import Numeric.LinearAlgebra qualified as LA
import Numeric.LinearAlgebra.Static qualified as H

type MLP i h o =
    Network
        '[FullyConnected i h, Tanh, FullyConnected h o]
        '[ 'D1 i, 'D1 h, 'D1 h, 'D1 o]

type Layers i h o = '[FullyConnected i h, Tanh, FullyConnected h o]

data Batch i = Batch
    { inputs :: [S ('D1 i)]
    , targets :: [Int]
    }

main :: IO ()
main = do
    args <- getArgs
    case args of
        [directory, count] | Just epochs <- readMaybe count ->
            runIris (experiment @4 @16 @3) directory epochs
        _ -> fail "Usage: grenade-iris DATA_DIRECTORY EPOCHS"

experiment
    :: forall i h o
     . (KnownNat i, KnownNat h, KnownNat o)
    => Experiment (MLP i h o) (Batch i)
experiment =
    Experiment
        { framework = "grenade"
        , executionNotes =
            "Per-sample forward/backward, mean gradient accumulated once per shared minibatch; SGD"
        , revision = "0.1.0@83cb4e4ec27ece8e7a7540d096bcf6c5ef3e8e7d"
        , loadParameters = \directory -> do
            tokens <- words <$> readFile (directory </> "weights.txt")
            values <- maybe (fail "Invalid initial weights") pure $ traverse readMaybe tokens
            let ni = size @i
                nh = size @h
                no = size @o
                (w1, rest) = splitAt (ni * nh) values
                (b1, remaining) = splitAt nh rest
                (w2, b2) = splitAt (nh * no) remaining
            unless (length values == ni * nh + nh + nh * no + no) $ fail "Wrong initial parameter count"
            let model = dense @i @h w1 b1 :~> Tanh :~> dense @h @o w2 b2 :~> NNil
            _ <- evaluate $ sum $ modelValues model
            pure model
        , prepareBatch = \features labels -> do
            let vectors = map (S1D . staticVector @i) features
            _ <- evaluate $ sum $ concatMap (\(S1D x) -> LA.toList $ H.extract x) vectors
            _ <- evaluate $ sum labels
            pure $ Batch vectors labels
        , parameterValues = evaluate . force . modelValues
        , predict = \model batch ->
            evaluate $ force $ concatMap (outputValues . snd . runNetwork model) $ inputs batch
        , trainBatch = updateBatch
        , validateBatch = \model batch -> do
            let examples = inputs batch
                labels = targets batch
                outputs = map (outputValues . snd . runNetwork model) examples
                losses = zipWith (\scores label -> fst $ crossEntropy scores label) outputs labels
                correct scores label = if snd (maximum $ zip scores [0 :: Int ..]) == label then 1 else 0
                accuracy = sum (zipWith correct outputs labels) / fromIntegral (length labels)
            evaluate $ force (sum losses / fromIntegral (length labels), accuracy)
        }

size :: forall n. KnownNat n => Int
size = fromIntegral $ natVal $ Proxy @n

staticVector :: forall n. KnownNat n => [Double] -> H.R n
staticVector values = fromMaybe (error "Incorrect vector size") $ H.create $ LA.fromList values

dense
    :: forall i o
     . (KnownNat i, KnownNat o)
    => [Double]
    -> [Double]
    -> FullyConnected i o
dense weights bias =
    FullyConnected
        (FullyConnected' (staticVector bias) matrix)
        (FullyConnected' (H.konst 0) (H.konst 0))
  where
    matrix =
        fromMaybe (error "Incorrect matrix size") $
            H.create $
                LA.tr $
                    LA.reshape (size @o) $
                        LA.fromList weights

outputValues :: S ('D1 n) -> [Double]
outputValues (S1D values) = LA.toList $ H.extract values

modelValues :: (KnownNat i, KnownNat h, KnownNat o) => MLP i h o -> [Double]
modelValues
    (FullyConnected (FullyConnected' b1 w1) _ :~> Tanh :~> FullyConnected (FullyConnected' b2 w2) _ :~> NNil) =
        matrixValues w1 ++ LA.toList (H.extract b1) ++ matrixValues w2 ++ LA.toList (H.extract b2)
      where
        matrixValues :: (KnownNat n, KnownNat m) => H.L n m -> [Double]
        matrixValues = LA.toList . LA.flatten . LA.tr . H.extract

modelTotal :: (KnownNat i, KnownNat h, KnownNat o) => MLP i h o -> Double
modelTotal
    (FullyConnected (FullyConnected' b1 w1) _ :~> Tanh :~> FullyConnected (FullyConnected' b2 w2) _ :~> NNil) =
        LA.sumElements (H.extract b1)
            + LA.sumElements (H.extract w1)
            + LA.sumElements (H.extract b2)
            + LA.sumElements (H.extract w2)

crossEntropy :: [Double] -> Int -> (Double, [Double])
crossEntropy scores target =
    let maximumScore = maximum scores
        exponentials = map (exp . subtract maximumScore) scores
        total = sum exponentials
        probabilities = map (/ total) exponentials
     in ( log total + maximumScore - scores !! target
        , zipWith (\p index -> p - if index == target then 1 else 0) probabilities [0 ..]
        )

updateBatch
    :: forall i h o
     . (KnownNat i, KnownNat h, KnownNat o)
    => MLP i h o
    -> Batch i
    -> IO (Double, MLP i h o)
updateBatch model (Batch examples labels) = do
    let count = fromIntegral $ length examples
        one input label =
            let (tape, output) = runNetwork model input
                (sampleLoss, seed) = crossEntropy (outputValues output) label
                (gradient, _) = runGradient model tape $ S1D $ staticVector @o $ map (/ count) seed
             in (sampleLoss / count, gradient)
        results = zipWith one examples labels
        gradients = foldl1 addGradients $ map snd results
        updated = applyUpdate (LearningParameters 0.1 0 0) model gradients
        loss = sum $ map fst results
    _ <- evaluate loss
    _ <- evaluate $ modelTotal updated
    pure (loss, updated)

addGradients
    :: (KnownNat i, KnownNat h, KnownNat o)
    => Gradients (Layers i h o)
    -> Gradients (Layers i h o)
    -> Gradients (Layers i h o)
addGradients
    (FullyConnected' b1 w1 :/> (() :/> (FullyConnected' b2 w2 :/> GNil)))
    (FullyConnected' c1 v1 :/> (() :/> (FullyConnected' c2 v2 :/> GNil))) =
        FullyConnected' (b1 + c1) (w1 + v1) :/> (() :/> (FullyConnected' (b2 + c2) (w2 + v2) :/> GNil))

data Experiment model batch = Experiment
    { framework :: String
    , revision :: String
    , executionNotes :: String
    , loadParameters :: FilePath -> IO model
    , prepareBatch :: [[Double]] -> [Int] -> IO batch
    , parameterValues :: model -> IO [Double]
    , predict :: model -> batch -> IO [Double]
    , trainBatch :: model -> batch -> IO (Double, model)
    , validateBatch :: model -> batch -> IO (Double, Double)
    }

runIris :: Experiment model batch -> FilePath -> Int -> IO ()
runIris spec directory epochs = do
    unless (epochs > 0) $ fail "Epochs must be positive"
    training <- loadBatch "train.csv" 120
    testing <- loadBatch "test.csv" 30
    initial <- loadParameters spec directory
    setupStart <- seconds
    logits <- predict spec initial training
    _ <- evaluate $ sum logits
    setupEnd <- seconds
    (firstLoss, firstParameters) <- update initial training
    firstValues <- parameterValues spec firstParameters
    _ <- evaluate $ sum firstValues
    firstEnd <- seconds
    (finalParameters, losses) <- foldM
        (\(parameters, history) _ -> do
            (loss, next) <- update parameters training
            pure (next, loss : history))
        (firstParameters, [firstLoss])
        [2 .. epochs]
    trainingEnd <- seconds
    (testLoss, accuracy) <- validateBatch spec finalParameters testing
    putStrLn $ JSON.encode $ JSON.makeObj
        [ ("framework", JSON.showJSON $ framework spec)
        , ("version", JSON.showJSON $ revision spec)
        , ("compiler", JSON.showJSON $ compilerName ++ "-" ++ showVersion compilerVersion)
        , ("execution_notes", JSON.showJSON $ executionNotes spec)
        , ("task", JSON.showJSON ("iris-mlp" :: String))
        , ("device", JSON.showJSON ("cpu" :: String))
        , ("dtype", JSON.showJSON ("float64" :: String))
        , ("epochs", JSON.showJSON epochs)
        , ("setup_seconds", JSON.showJSON $ setupEnd - setupStart)
        , ("first_step_seconds", JSON.showJSON $ firstEnd - setupEnd)
        , ("training_seconds", JSON.showJSON $ trainingEnd - setupEnd)
        , ("initial_logits", JSON.showJSON logits)
        , ("first_step_parameters", JSON.showJSON firstValues)
        , ("losses", JSON.showJSON $ reverse losses)
        , ("test_loss", JSON.showJSON testLoss)
        , ("test_accuracy", JSON.showJSON accuracy)
        ]
  where
    loadBatch name count = do
        rows <- lines <$> readFile (directory </> name)
        samples <- mapM parseRow rows
        unless (length samples == count) $ fail "Unexpected Iris sample count"
        prepareBatch spec (map fst samples) (map snd samples)
    update parameters batch = do
        (loss, next) <- trainBatch spec parameters batch
        when (isNaN loss || isInfinite loss) $ fail "Nonfinite loss"
        pure (loss, next)
    parseRow row = case words $ map (\c -> if c == ',' then ' ' else c) row of
        [a, b, c, d, label] -> case (traverse readMaybe [a, b, c, d], readMaybe label) of
            (Just features, Just target) | target >= 0 && target < 3 -> pure (features, target)
            _ -> fail "Invalid Iris row"
        _ -> fail "Expected four features and a class"

seconds :: IO Double
seconds = (/ 1e9) . fromIntegral <$> getMonotonicTimeNSec
