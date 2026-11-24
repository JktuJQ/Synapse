{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Main (main) where

import Control.Exception (evaluate)
import Control.Monad (foldM, forM, unless, when)
import Control.Monad.Trans.State.Strict (get, put)
import Data.ByteString qualified as BS
import Data.Word (Word32)
import GHC.Clock (getMonotonicTimeNSec)
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat)
import Synapse.Autograd qualified as AG
import Synapse.NN qualified as NN
import Synapse.Tensor qualified as T
import System.Environment (getArgs)
import System.FilePath ((</>))
import Text.JSON qualified as JSON
import Text.Read (readMaybe)

data Parameters t = Parameters
    { hidden :: NN.Dense 784 64 t Double
    , output :: NN.Dense 64 10 t Double
    }
    deriving (Generic)

instance T.TensorStructure (Parameters T.Tensor) where
    type WithTensor t (Parameters T.Tensor) = Parameters t

predict
    :: (KnownNat batch, T.RealFloatTensorOps t Double)
    => Parameters t
    -> t '[batch, 784] Double
    -> t '[batch, 10] Double
predict parameters = NN.dense (output parameters) . NN.tanh . NN.dense (hidden parameters)

trainingStep
    :: KnownNat batch
    => T.Tensor '[batch, 784] Double
    -> T.Tensor '[batch] Int
    -> NN.State (Parameters T.Tensor) (T.Tensor '[] Double)
trainingStep features targets = do
    parameters <- get
    let objective
            :: AG.Context s
            => Parameters (AG.Tensor s)
            -> AG.Tensor s '[] Double
        objective p = NN.unsafeCrossEntropy @1 T.mean (predict p $ AG.constant features) targets
        (loss, gradients) = AG.backprop objective parameters
    put $ NN.evalState (NN.optimizerStep (NN.SGD 0.1) parameters gradients) ()
    pure loss

experiment :: Experiment (Parameters T.TensorData)
experiment =
    Experiment
        { modelName = "mlp"
        , initialParameters = \directory -> do
            values <- readWeights (directory </> "weights-mlp.txt") 50890
            let (first, rest) = splitAt 50176 values
                (firstBias, remaining) = splitAt 64 rest
                (second, secondBias) = splitAt 640 remaining
            pure $
                Parameters
                    (NN.Dense (T.fromListData first) $ T.fromListData firstBias)
                    (NN.Dense (T.fromListData second) $ T.fromListData secondBias)
        , parameterValues = \(Parameters (NN.Dense first firstBias) (NN.Dense second secondBias)) ->
            T.toList first ++ T.toList firstBias ++ T.toList second ++ T.toList secondBias
        , infer = T.run T.defaultCPU predict
        , step = \parameters batch -> preparedStep parameters (features batch) (targets batch)
        , validate = \parameters batch -> preparedValidation parameters (features batch) (targets batch)
        }
  where
    preparedStep = T.run T.defaultCPU $ \parameters features targets ->
        NN.runState (trainingStep features targets) parameters
    preparedValidation = T.run T.defaultCPU $ \parameters features targets ->
        let logits = predict parameters features
         in (NN.unsafeCrossEntropy @1 T.mean logits targets, NN.accuracy @1 logits targets)

main :: IO ()
main = runExperiment experiment

data Batch = Batch
    { features :: T.TensorData '[100, 784] Double
    , targets :: T.TensorData '[100] Int
    }

readBatches
    :: FilePath
    -> String
    -> IO [Batch]
readBatches directory split = do
    images <- BS.readFile $ directory </> split ++ "-images.idx"
    labels <- BS.readFile $ directory </> split ++ "-labels.idx"
    unless (BS.length images >= 16 && BS.length labels >= 8) $ fail "Truncated IDX headers"
    let count = fromIntegral $ word32 labels 4
    unless
        ( word32 images 0 == 2051
            && word32 labels 0 == 2049
            && word32 images 4 == fromIntegral count
            && word32 images 8 == 28
            && word32 images 12 == 28
            && count > 0
            && count `mod` 100 == 0
            && BS.length images == 16 + count * 784
            && BS.length labels == 8 + count
            && BS.all (< 10) (BS.drop 8 labels)
        )
        $ fail "Expected matching 28x28 IDX images and labels, in complete batches of 100"
    forM [0, 100 .. count - 100] $ \offset -> do
        let pixels = BS.take 78400 $ BS.drop (16 + offset * 784) images
            classes = BS.take 100 $ BS.drop (8 + offset) labels
            input = T.fromListData $ map ((/ 255) . fromIntegral) $ BS.unpack pixels
            output = T.fromListData $ map fromIntegral $ BS.unpack classes
        _ <- evaluate $ sum $ T.toList input
        _ <- evaluate $ sum $ T.toList output
        pure $ Batch input output

word32
    :: BS.ByteString
    -> Int
    -> Word32
word32 bytes offset =
    BS.foldl' (\value byte -> value * 256 + fromIntegral byte) 0 $ BS.take 4 $ BS.drop offset bytes

readWeights
    :: FilePath
    -> Int
    -> IO [Double]
readWeights path count = do
    tokens <- words <$> readFile path
    values <- maybe (fail $ path ++ ": invalid weight") pure $ traverse readMaybe tokens
    unless (length values == count && all (\x -> not (isNaN x || isInfinite x)) values) $
        fail $
            path ++ ": unexpected parameter count or nonfinite weight"
    _ <- evaluate $ sum values
    pure values

data Experiment parameters = Experiment
    { modelName :: String
    , initialParameters :: FilePath -> IO parameters
    , parameterValues :: parameters -> [Double]
    , infer :: parameters -> T.TensorData '[100, 784] Double -> T.TensorData '[100, 10] Double
    , step :: parameters -> Batch -> (T.TensorData '[] Double, parameters)
    , validate :: parameters -> Batch -> (T.TensorData '[] Double, T.TensorData '[] Double)
    }

seconds :: IO Double
seconds = (/ 1e9) . fromIntegral <$> getMonotonicTimeNSec

runExperiment :: Experiment parameters -> IO ()
runExperiment spec = do
    args <- getArgs
    case args of
        [directory, count] | Just epochs <- readMaybe count -> runTraining spec directory epochs
        _ -> fail "Usage: DATA_DIRECTORY EPOCHS"

runTraining
    :: Experiment parameters
    -> FilePath
    -> Int
    -> IO ()
runTraining spec directory epochs = do
    unless (epochs > 0) $ fail "Epoch count must be positive"
    training <- readBatches directory "train"
    testing <- readBatches directory "test"
    parameters <- initialParameters spec directory
    first <- case training of
        batch : _ -> pure batch
        [] -> fail "Empty training dataset"
    setupStart <- seconds
    let logits = T.toList $ infer spec parameters $ features first
    _ <- evaluate $ sum logits
    setupEnd <- seconds
    (firstLoss, firstParameters) <- update parameters first
    let firstValues = parameterValues spec firstParameters
    _ <- evaluate $ sum firstValues
    firstEnd <- seconds
    (afterFirstEpoch, firstHistory) <- train firstParameters [firstLoss] $ drop 1 training
    (finalParameters, histories) <-
        foldM
            ( \(p, history) _ -> do
                (next, losses) <- train p [] training
                pure (next, reverse losses : history)
            )
            (afterFirstEpoch, [reverse firstHistory])
            [2 .. epochs]
    trainingEnd <- seconds
    measurements <- mapM (measure finalParameters) testing
    let average = (/ fromIntegral (length measurements)) . sum
        losses = concat $ reverse histories
        epochLosses = map ((/ fromIntegral (length training)) . sum) $ reverse histories
    putStrLn $
        JSON.encode $
            JSON.makeObj
                [ ("framework", JSON.showJSON ("synapse" :: String))
                , ("task", JSON.showJSON $ "mnist-" ++ modelName spec)
                , ("device", JSON.showJSON ("cpu" :: String))
                , ("dtype", JSON.showJSON ("float64" :: String))
                , ("pooling", JSON.showJSON $ if modelName spec == "cnn" then "max" else ("none" :: String))
                , ("epochs", JSON.showJSON epochs)
                , ("train_count", JSON.showJSON $ 100 * length training)
                , ("test_count", JSON.showJSON $ 100 * length testing)
                , ("setup_seconds", JSON.showJSON $ setupEnd - setupStart)
                , ("first_step_seconds", JSON.showJSON $ firstEnd - setupEnd)
                , ("training_seconds", JSON.showJSON $ trainingEnd - setupEnd)
                , ("initial_logits", JSON.showJSON logits)
                , ("first_step_parameters", JSON.showJSON firstValues)
                , ("losses", JSON.showJSON losses)
                , ("epoch_losses", JSON.showJSON epochLosses)
                , ("test_loss", JSON.showJSON $ average $ map fst measurements)
                , ("test_accuracy", JSON.showJSON $ average $ map snd measurements)
                ]
  where
    update parameters batch = do
        let (loss, next) = step spec parameters batch
        value <- evaluate $ sum $ T.toList loss
        when (isNaN value || isInfinite value) $ fail "Nonfinite training loss"
        pure (value, next)
    train parameters history =
        foldM
            ( \(p, losses) batch -> do
                (loss, next) <- update p batch
                pure (next, loss : losses)
            )
            (parameters, history)
    measure parameters batch = do
        let (loss, accuracy) = validate spec parameters batch
        value <- evaluate $ sum $ T.toList loss
        score <- evaluate $ sum $ T.toList accuracy
        pure (value, score)
