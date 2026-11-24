{-# LANGUAGE AllowAmbiguousTypes #-}

module Main (main) where

import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Monad (foldM, forM, unless, when)
import Data.ByteString qualified as BS
import Data.Maybe (fromMaybe)
import Data.Proxy (Proxy (..))
import Data.Version (showVersion)
import GHC.Clock (getMonotonicTimeNSec)
import GHC.TypeLits (KnownNat, natVal)
import Grenade hiding (train)
import Numeric.LinearAlgebra qualified as LA
import Numeric.LinearAlgebra.Static qualified as H
import System.Environment (getArgs)
import System.FilePath ((</>))
import System.Info (compilerName, compilerVersion)
import Text.JSON qualified as JSON
import Text.Read (readMaybe)

type MLP i h o =
    Network
        '[FullyConnected i h, Tanh, FullyConnected h o]
        '[ 'D1 i, 'D1 h, 'D1 h, 'D1 o]

type Layers i h o = '[FullyConnected i h, Tanh, FullyConnected h o]

data Batch i = Batch
    { inputs :: [S ('D1 i)]
    , targets :: [Int]
    }

experiment
    :: forall i h o
     . (KnownNat i, KnownNat h, KnownNat o)
    => FilePath
    -> Experiment (MLP i h o) (Batch i)
experiment file =
    Experiment
        { framework = "grenade"
        , executionNotes =
            "Per-sample forward/backward, mean gradient accumulated once per shared minibatch; SGD"
        , revision = "0.1.0@83cb4e4ec27ece8e7a7540d096bcf6c5ef3e8e7d"
        , loadParameters = \directory -> do
            tokens <- words <$> readFile (directory </> file)
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

main :: IO ()
main = runComparison "mnist-mlp" (experiment @784 @64 @10 "weights-mlp.txt")

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

crossEntropy :: [Double] -> Int -> (Double, [Double])
crossEntropy scores target =
    let maximumScore = maximum scores
        exponentials = map (exp . subtract maximumScore) scores
        total = sum exponentials
        probabilities = map (/ total) exponentials
     in ( log total + maximumScore - scores !! target
        , zipWith (\p index -> p - if index == target then 1 else 0) probabilities [0 ..]
        )

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

runComparison :: String -> Experiment model batch -> IO ()
runComparison task spec = do
    args <- getArgs
    case args of
        [directory, count] | Just epochs <- readMaybe count -> runTask spec directory task epochs
        _ -> fail "Usage: DATA_DIRECTORY EPOCHS"

runTask
    :: Experiment model batch
    -> FilePath
    -> String
    -> Int
    -> IO ()
runTask spec directory task epochs = do
    unless (epochs > 0) $ fail "Epochs must be positive"
    training <- loadBatches spec directory True
    testing <- loadBatches spec directory False
    initial <- loadParameters spec directory
    first <- case training of
        x : _ -> pure x
        [] -> fail "Empty training dataset"
    setupStart <- seconds
    logits <- predict spec initial first
    _ <- evaluate $ sum logits
    setupEnd <- seconds
    (firstLoss, firstParameters) <- update initial first
    firstValues <- parameterValues spec firstParameters
    _ <- evaluate $ sum firstValues
    firstEnd <- seconds
    (afterFirst, firstHistory) <- train firstParameters [firstLoss] $ drop 1 training
    (finalParameters, histories) <-
        foldM
            ( \(parameters, history) _ -> do
                (next, losses) <- train parameters [] training
                pure (next, reverse losses : history)
            )
            (afterFirst, [reverse firstHistory])
            [2 .. epochs]
    trainingEnd <- seconds
    measurements <- mapM (validateBatch spec finalParameters) testing
    let mean values = sum values / fromIntegral (length values)
        orderedHistory = reverse histories
        batchSize = 100
        fields =
            [ ("framework", JSON.showJSON $ framework spec)
            , ("version", JSON.showJSON $ revision spec)
            , ("compiler", JSON.showJSON $ compilerName ++ "-" ++ showVersion compilerVersion)
            , ("execution_notes", JSON.showJSON $ executionNotes spec)
            , ("task", JSON.showJSON $ task)
            , ("device", JSON.showJSON ("cpu" :: String))
            , ("dtype", JSON.showJSON ("float64" :: String))
            , ("pooling", JSON.showJSON $ if task == "mnist-cnn" then "max" else ("none" :: String))
            , ("epochs", JSON.showJSON epochs)
            , ("train_count", JSON.showJSON $ batchSize * length training)
            , ("test_count", JSON.showJSON $ batchSize * length testing)
            , ("setup_seconds", JSON.showJSON $ setupEnd - setupStart)
            , ("first_step_seconds", JSON.showJSON $ firstEnd - setupEnd)
            , ("training_seconds", JSON.showJSON $ trainingEnd - setupEnd)
            , ("initial_logits", JSON.showJSON logits)
            , ("first_step_parameters", JSON.showJSON firstValues)
            , ("losses", JSON.showJSON $ concat orderedHistory)
            , ("test_loss", JSON.showJSON $ mean $ map fst measurements)
            , ("test_accuracy", JSON.showJSON $ mean $ map snd measurements)
            ]
        epochFields = [("epoch_losses", JSON.showJSON $ map mean orderedHistory)]
    putStrLn $ JSON.encode $ JSON.makeObj $ fields ++ epochFields
  where
    update parameters batch = do
        (loss, next) <- trainBatch spec parameters batch
        _ <- evaluate loss
        whenNonfinite loss
        pure (loss, next)
    train parameters history =
        foldM
            ( \(p, losses) batch -> do
                (loss, next) <- update p batch
                pure (next, loss : losses)
            )
            (parameters, history)
    whenNonfinite value = when (isNaN value || isInfinite value) $ fail "Nonfinite loss"

seconds :: IO Double
seconds = (/ 1e9) . fromIntegral <$> getMonotonicTimeNSec

loadBatches
    :: Experiment model batch
    -> FilePath
    -> Bool
    -> IO [batch]
loadBatches spec directory training = do
    let prefix = if training then "train" else "test"
    images <- BS.readFile $ directory </> prefix ++ "-images.idx"
    labels <- BS.readFile $ directory </> prefix ++ "-labels.idx"
    unless (BS.length images >= 16 && BS.length labels >= 8) $ fail "Truncated IDX headers"
    let word :: Int -> BS.ByteString -> Int
        word offset bytes =
            BS.foldl' (\n byte -> 256 * n + fromIntegral byte) 0 $
                BS.take 4 $
                    BS.drop offset bytes
        count = word 4 labels
    unless
        ( word 0 images == 2051
            && word 0 labels == 2049
            && word 4 images == count
            && word 8 images == 28
            && word 12 images == 28
            && count > 0
            && count `mod` 100 == 0
            && BS.length images == 16 + count * 784
            && BS.length labels == 8 + count
            && BS.all (< 10) (BS.drop 8 labels)
        )
        $ fail "Invalid MNIST IDX data"
    forM [0, 100 .. count - 100] $ \offset -> do
        let pixels index =
                map ((/ 255) . fromIntegral) $
                    BS.unpack $
                        BS.take 784 $
                            BS.drop (16 + index * 784) images
            classes = map fromIntegral $ BS.unpack $ BS.take 100 $ BS.drop (8 + offset) labels
        prepareBatch spec (map pixels [offset .. offset + 99]) classes
