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
    { embedding :: NN.Dense 16 16 t Double
    , selfAttention :: NN.MultiHeadAttention 16 2 t Double
    , ff1 :: NN.Dense 16 32 t Double
    , ff2 :: NN.Dense 32 16 t Double
    , classifier :: NN.Dense 16 10 t Double
    , norm1 :: NN.LayerNorm '[16] t Double
    , norm2 :: NN.LayerNorm '[16] t Double
    }
    deriving (Generic)

instance T.TensorStructure (Parameters T.Tensor) where
    type WithTensor t (Parameters T.Tensor) = Parameters t

predict
    :: forall batch t
     . (KnownNat batch, T.RealFloatTensorOps t Double)
    => Parameters t -> t '[batch, 784] Double -> t '[batch, 10] Double
predict p input = NN.dense (classifier p) $ T.meanAxes @'[1] final
  where
    patches =
        T.reshape @'[batch, 49, 16] $
            T.permuteAxes @'[0, 1, 3, 2, 4] $
                T.reshape @'[batch, 7, 4, 7, 4] input
    embedded = NN.dense (embedding p) patches
    x = embedded + T.broadcast (NN.sinusoidalPositions @49 @16 1)
    z = NN.layerNorm 1e-5 (norm1 p) x
    residual = x + NN.multiHeadAttention (selfAttention p) Nothing (z, z, z)
    normalized = NN.layerNorm 1e-5 (norm2 p) residual
    final = residual + NN.dense (ff2 p) (NN.tanh $ NN.dense (ff1 p) normalized)

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
        { modelName = "vit"
        , initialParameters = \directory -> do
            values <- readWeights (directory </> "weights-vit.txt") 2666
            let (embeddingW, rest0) = splitAt 256 values
                (embeddingB, rest1) = splitAt 16 rest0
                (qW, rest2) = splitAt 256 rest1
                (qB, rest3) = splitAt 16 rest2
                (kW, rest4) = splitAt 256 rest3
                (kB, rest5) = splitAt 16 rest4
                (vW, rest6) = splitAt 256 rest5
                (vB, rest7) = splitAt 16 rest6
                (projectionW, rest8) = splitAt 256 rest7
                (projectionB, rest9) = splitAt 16 rest8
                (ff1W, rest10) = splitAt 512 rest9
                (ff1B, rest11) = splitAt 32 rest10
                (ff2W, rest12) = splitAt 512 rest11
                (ff2B, rest13) = splitAt 16 rest12
                (classifierW, rest14) = splitAt 160 rest13
                (classifierB, rest15) = splitAt 10 rest14
                (n1w, rest16) = splitAt 16 rest15
                (n1b, rest17) = splitAt 16 rest16
                (n2w, rest18) = splitAt 16 rest17
                n2b = rest18
            pure $
                Parameters
                    (NN.Dense (T.fromListData embeddingW) (T.fromListData embeddingB))
                    ( NN.MultiHeadAttention
                        (NN.Dense (T.fromListData qW) (T.fromListData qB))
                        (NN.Dense (T.fromListData kW) (T.fromListData kB))
                        (NN.Dense (T.fromListData vW) (T.fromListData vB))
                        (NN.Dense (T.fromListData projectionW) (T.fromListData projectionB))
                    )
                    (NN.Dense (T.fromListData ff1W) (T.fromListData ff1B))
                    (NN.Dense (T.fromListData ff2W) (T.fromListData ff2B))
                    (NN.Dense (T.fromListData classifierW) (T.fromListData classifierB))
                    (NN.LayerNorm (T.fromListData n1w) (T.fromListData n1b))
                    (NN.LayerNorm (T.fromListData n2w) (T.fromListData n2b))
        , parameterValues = \p ->
            concat
                [ T.toList (NN.denseWeights (embedding p))
                , T.toList (NN.denseBias (embedding p))
                , T.toList (NN.denseWeights (NN.attentionQuery (selfAttention p)))
                , T.toList (NN.denseBias (NN.attentionQuery (selfAttention p)))
                , T.toList (NN.denseWeights (NN.attentionKey (selfAttention p)))
                , T.toList (NN.denseBias (NN.attentionKey (selfAttention p)))
                , T.toList (NN.denseWeights (NN.attentionValue (selfAttention p)))
                , T.toList (NN.denseBias (NN.attentionValue (selfAttention p)))
                , T.toList (NN.denseWeights (NN.attentionOutput (selfAttention p)))
                , T.toList (NN.denseBias (NN.attentionOutput (selfAttention p)))
                , T.toList (NN.denseWeights (ff1 p))
                , T.toList (NN.denseBias (ff1 p))
                , T.toList (NN.denseWeights (ff2 p))
                , T.toList (NN.denseBias (ff2 p))
                , T.toList (NN.denseWeights (classifier p))
                , T.toList (NN.denseBias (classifier p))
                , T.toList (NN.layerNormWeights (norm1 p))
                , T.toList (NN.layerNormBias (norm1 p))
                , T.toList (NN.layerNormWeights (norm2 p))
                , T.toList (NN.layerNormBias (norm2 p))
                ]
        , infer = T.run2 T.CPU predict
        , step = \parameters batch -> preparedStep parameters (features batch) (targets batch)
        , validate = \parameters batch -> preparedValidation parameters (features batch) (targets batch)
        }
  where
    preparedStep = T.run3 T.CPU $ \parameters features targets ->
        NN.runState (trainingStep features targets) parameters
    preparedValidation = T.run3 T.CPU $ \parameters features targets ->
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
