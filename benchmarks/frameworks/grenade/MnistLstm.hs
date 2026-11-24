module Main (main) where

import Control.Exception (evaluate)
import Control.Monad (foldM, unless)
import Control.Monad.Random (evalRand)
import Data.List.NonEmpty (NonEmpty (..))
import GHC.Clock (getMonotonicTimeNSec)
import GHC.TypeNats (KnownNat)
import System.Environment (getArgs)
import System.FilePath ((</>))
import System.Random (mkStdGen)
import Text.Read (readMaybe)

import Data.ByteString qualified as BS
import Grenade
import Grenade.Recurrent (RecurrentLayer (..))
import Grenade.Recurrent.Layers.LSTM (LSTM (..), LSTMWeights (..))
import Numeric.LinearAlgebra qualified as LA
import Numeric.LinearAlgebra.Static qualified as H
import Text.JSON qualified as JSON

type Model = (LSTM 28 16, FullyConnected 16 10)
type Sample = (NonEmpty (S ('D1 28)), Int)
type StepTape = RecTape (LSTM 28 16) ('D1 28) ('D1 16)
type GradientsForSample = (LSTMWeights 28 16, FullyConnected' 16 10)

main :: IO ()
main = do
    args <- getArgs
    case args of
        [directory, target, limit]
            | Just accuracy <- readMaybe target
            , Just epochs <- readMaybe limit
            , accuracy > 0 && accuracy <= 1 && epochs > 0 ->
                benchmark directory accuracy epochs
        _ -> fail "Usage: grenade-mnist-lstm DATA_DIRECTORY TARGET_ACCURACY MAX_EPOCHS"

benchmark :: FilePath -> Double -> Int -> IO ()
benchmark directory target limit = do
    training <- readSamples directory "train"
    testing <- readSamples directory "test"
    let initial = evalRand ((,) <$> createRandom <*> createRandom) (mkStdGen 17)
    _ <- evaluate $ modelTotal initial
    start <- seconds
    (epoch, accuracy, trainingTime, evaluationTime) <- trainUntil training testing target limit initial
    elapsed <- subtract start <$> seconds
    let reached = accuracy >= target
    putStrLn $
        JSON.encode $
            JSON.makeObj
                [ ("framework", JSON.showJSON ("grenade" :: String))
                , ("task", JSON.showJSON ("mnist-lstm" :: String))
                , ("architecture", JSON.showJSON ("peephole-lstm" :: String))
                , ("target_accuracy", JSON.showJSON target)
                , ("test_accuracy", JSON.showJSON accuracy)
                , ("target_reached", JSON.showJSON reached)
                , ("epochs", JSON.showJSON epoch)
                , ("max_epochs", JSON.showJSON limit)
                , ("steps", JSON.showJSON $ epoch * (length training `div` 100))
                , ("train_count", JSON.showJSON $ length training)
                , ("test_count", JSON.showJSON $ length testing)
                , ("training_seconds", JSON.showJSON trainingTime)
                , ("evaluation_seconds", JSON.showJSON evaluationTime)
                , ("elapsed_seconds", JSON.showJSON elapsed)
                , ("time_to_target_seconds", if reached then JSON.showJSON elapsed else JSON.JSNull)
                ]

trainUntil :: [Sample] -> [Sample] -> Double -> Int -> Model -> IO (Int, Double, Double, Double)
trainUntil training testing target limit = go 1 0 0
  where
    batches = chunks 100 training
    go epoch !trainingTime !evaluationTime model = do
        start <- seconds
        updated <- foldM update model batches
        afterTraining <- seconds
        let correct (image, label) =
                let scores = classify updated image
                 in snd (maximum $ zip (values scores) [0 ..]) == label
        accuracy <- evaluate $ fromIntegral (length $ filter correct testing) / fromIntegral (length testing)
        afterEvaluation <- seconds
        let nextTraining = trainingTime + (afterTraining - start)
            nextEvaluation = evaluationTime + (afterEvaluation - afterTraining)
        if accuracy >= target || epoch >= limit
            then pure (epoch, accuracy, nextTraining, nextEvaluation)
            else go (epoch + 1) nextTraining nextEvaluation updated

update :: Model -> [Sample] -> IO Model
update model@(recurrent, classifier) samples = do
    let scale = 1 / fromIntegral (length samples)
        (recurrentGradient, classifierGradient) =
            foldl' (\total sample -> addGradients total (sampleGradient model scale sample)) zeroGradients samples
        settings = LearningParameters 0.1 0 0
        next = (runUpdate settings recurrent recurrentGradient, runUpdate settings classifier classifierGradient)
    _ <- evaluate $ modelTotal next
    pure next

forwardStep :: LSTM 28 16 -> S ('D1 16) -> S ('D1 28) -> (StepTape, S ('D1 16), S ('D1 16))
forwardStep = runRecurrentForwards

classify :: Model -> NonEmpty (S ('D1 28)) -> S ('D1 10)
classify (recurrent, classifier) (first :| remaining) =
    let (_, cell, hidden) = forwardStep recurrent 0 first
        advance (!memory, _) row =
            let (_, nextMemory, nextHidden) = forwardStep recurrent memory row
             in (nextMemory, nextHidden)
        (_, finalHidden) = foldl' advance (cell, hidden) remaining
     in snd $ runForwards classifier finalHidden

sampleGradient :: Model -> Double -> Sample -> GradientsForSample
sampleGradient (recurrent, classifier) scale (first :| remaining, target) =
    let (firstTape, firstCell, firstHidden) = forwardStep recurrent 0 first
        advance (previousTapes, !cell, _) row =
            let (tape, nextCell, nextHidden) = forwardStep recurrent cell row
             in (tape : previousTapes, nextCell, nextHidden)
        (tapes, _, hidden) = foldl' advance ([firstTape], firstCell, firstHidden) remaining
        (classifierTape, scores) = runForwards @(FullyConnected 16 10) @('D1 16) @('D1 10) classifier hidden
        logits = values scores
        exponentials = map (exp . subtract (maximum logits)) logits
        normalizer = sum exponentials
        seed =
            vector $
                zipWith
                    (\index e -> scale * (e / normalizer - if index == target then 1 else 0))
                    [0 ..]
                    exponentials
        (classifierGradient, hiddenGradient) = runBackwards @(FullyConnected 16 10) @('D1 16) @('D1 10) classifier classifierTape seed
        back (total, !cellGradient, outputGradient) (tape :: StepTape) =
            let (gradient, nextCellGradient, _) = runRecurrentBackwards @(LSTM 28 16) @('D1 28) @('D1 16) recurrent tape cellGradient outputGradient
             in (addLstm total gradient, nextCellGradient, 0)
        (recurrentGradient, _, _) = foldl' back (zeroLstm, 0, hiddenGradient) tapes
     in (recurrentGradient, classifierGradient)

addGradients :: GradientsForSample -> GradientsForSample -> GradientsForSample
addGradients (!left, FullyConnected' bias weights) (!right, FullyConnected' db dw) =
    let !recurrent = addLstm left right
     in (recurrent, FullyConnected' (bias + db) (weights + dw))

addLstm :: LSTMWeights 28 16 -> LSTMWeights 28 16 -> LSTMWeights 28 16
addLstm
    (LSTMWeights wf uf bf wi ui bi wo uo bo wc bc)
    (LSTMWeights wf' uf' bf' wi' ui' bi' wo' uo' bo' wc' bc') =
        LSTMWeights
            (wf + wf')
            (uf + uf')
            (bf + bf')
            (wi + wi')
            (ui + ui')
            (bi + bi')
            (wo + wo')
            (uo + uo')
            (bo + bo')
            (wc + wc')
            (bc + bc')

zeroGradients :: GradientsForSample
zeroGradients = (zeroLstm, FullyConnected' 0 0)

zeroLstm :: LSTMWeights 28 16
zeroLstm = LSTMWeights 0 0 0 0 0 0 0 0 0 0 0

modelTotal :: Model -> Double
modelTotal (LSTM (LSTMWeights wf uf bf wi ui bi wo uo bo wc bc) _, FullyConnected (FullyConnected' b w) _) =
    sum $
        map
            LA.sumElements
            [ H.extract bf
            , H.extract bi
            , H.extract bo
            , H.extract bc
            , H.extract b
            , LA.flatten $ H.extract wf
            , LA.flatten $ H.extract uf
            , LA.flatten $ H.extract wi
            , LA.flatten $ H.extract ui
            , LA.flatten $ H.extract wo
            , LA.flatten $ H.extract uo
            , LA.flatten $ H.extract wc
            , LA.flatten $ H.extract w
            ]

values :: S ('D1 n) -> [Double]
values (S1D tensor) = LA.toList $ H.extract tensor

vector :: forall n. KnownNat n => [Double] -> S ('D1 n)
vector xs = S1D $ maybe (error "Invalid vector length") id $ H.create $ LA.fromList xs

readSamples :: FilePath -> String -> IO [Sample]
readSamples directory prefix = do
    images <- BS.readFile (directory </> prefix ++ "-images.idx")
    labels <- BS.readFile (directory </> prefix ++ "-labels.idx")
    let count = word32 labels 4
    unless
        ( word32 images 0 == 2051
            && word32 labels 0 == 2049
            && word32 images 4 == count
            && word32 images 8 == 28
            && word32 images 12 == 28
            && count > 0
            && count `mod` 100 == 0
            && BS.length images == 16 + count * 784
            && BS.length labels == 8 + count
            && BS.all (< 10) (BS.drop 8 labels)
        )
        $ fail "Invalid MNIST IDX data"
    let readRow image row =
            vector $
                map ((/ 255) . fromIntegral) $
                    BS.unpack $
                        BS.take 28 $
                            BS.drop (16 + 784 * image + 28 * row) images
        samples =
            [ (readRow i 0 :| map (readRow i) [1 .. 27], fromIntegral $ BS.index labels (8 + i))
            | i <- [0 .. count - 1]
            ]
    _ <- evaluate $ sum [sum (fmap (sum . values) rows) + fromIntegral label | (rows, label) <- samples]
    pure samples

word32 :: BS.ByteString -> Int -> Int
word32 bytes offset = BS.foldl' (\n byte -> 256 * n + fromIntegral byte) 0 $ BS.take 4 $ BS.drop offset bytes

chunks :: Int -> [a] -> [[a]]
chunks _ [] = []
chunks n xs = let (batch, remaining) = splitAt n xs in batch : chunks n remaining

seconds :: IO Double
seconds = (/ 1e9) . fromIntegral <$> getMonotonicTimeNSec
