module Main (main) where

import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Monad (foldM, forM, unless, void, when, zipWithM_)
import Data.ByteString qualified as BS
import Data.Version (showVersion)
import GHC.Clock (getMonotonicTimeNSec)
import System.Environment (getArgs)
import System.FilePath ((</>))
import System.Info (compilerName, compilerVersion)
import Text.JSON qualified as JSON
import Text.Read (readMaybe)
import Torch qualified
import Torch.Internal.Cast (cast2, cast3)
import Torch.Internal.Managed.Native.Native7 qualified as Native
import Torch.Internal.Managed.Type.Scalar (newScalar_d)
import Torch.Internal.Managed.Type.Tensor.Tensor0 (tensor_add__ts)

main :: IO ()
main = runModel "mlp" [[64, 784], [64], [10, 64], [10]] [0, 2] forward

forward :: [Torch.Tensor] -> Torch.Tensor -> Torch.Tensor
forward [w1, b1, w2, b2] input = dense w2 b2 $ Torch.tanh $ dense w1 b1 input
forward _ _ = error "Expected MLP parameters"

data Model = Model
    { parameters :: [Torch.Parameter]
    , detachedParameters :: [Torch.Tensor]
    }

data Batch = Batch
    { inputs :: Torch.Tensor
    , targets :: Torch.Tensor
    }

runModel
    :: String
    -> [[Int]]
    -> [Int]
    -> ([Torch.Tensor] -> Torch.Tensor -> Torch.Tensor)
    -> IO ()
runModel task shapes transposed forwardPass =
    runComparison ("mnist-" ++ task) $ experiment task shapes transposed forwardPass

experiment
    :: String
    -> [[Int]]
    -> [Int]
    -> ([Torch.Tensor] -> Torch.Tensor -> Torch.Tensor)
    -> Experiment Model Batch
experiment task shapes transposed forwardPass =
    Experiment
        { framework = "hasktorch"
        , executionNotes =
            "CPU LibTorch; explicit gradients; in-place SGD; RTS-managed GC"
        , revision =
            "0.2.2.0@059790c64e8556d8dd65978a2c38e50af84e6776"
        , loadParameters = \directory -> do
            tokens <- words <$> readFile (directory </> weightsFile)
            values <- maybe (fail "Invalid initial weights") pure $ traverse readMaybe tokens
            unless (length values == sum (map product shapes)) $ fail "Incorrect parameter count"
            tensors <- initialize values
            leaves <- mapM Torch.makeIndependent tensors
            detached <- mapM (Torch.detach . Torch.toDependent) leaves
            _ <- evaluate $ sum $ concatMap tensorValues tensors
            pure $ Model leaves detached
        , prepareBatch = \features labels -> do
            let x = Torch.toType Torch.Double $ Torch.asTensor features
                y = Torch.toType Torch.Int64 $ Torch.asTensor labels
            _ <- evaluate $ sum $ tensorValues x
            _ <- evaluate (Torch.asValue (Torch.sumAll y) :: Int)
            pure $ Batch x y
        , parameterValues = \model ->
            evaluate $ force $ concat $ zipWith canonical [0 :: Int ..] $ detachedParameters model
        , predict = \model batch ->
            evaluate $ force $ tensorValues $ forwardPass (detachedParameters model) $ inputs batch
        , trainBatch = \model batch -> do
            let logits = forwardPass (map Torch.toDependent $ parameters model) $ inputs batch
            loss <- crossEntropy logits $ targets batch
            value <- evaluate (Torch.asValue loss :: Double)
            gradients <- evaluate $ Torch.gradWithOptions (Torch.GradOptions False False False) loss (parameters model)
            alpha <- newScalar_d (-0.1)
            -- The detached tensors share storage with the gradient-tracked leaves.
            zipWithM_
                (\tensor gradient -> void (cast3 tensor_add__ts tensor gradient alpha :: IO Torch.Tensor))
                (detachedParameters model)
                gradients
            pure (value, model)
        , validateBatch = \model batch -> do
            let logits = forwardPass (detachedParameters model) $ inputs batch
                predictions = Torch.argmax (Torch.Dim 1) Torch.RemoveDim logits
                matches = Torch.toType Torch.Double $ Torch.eq predictions $ targets batch
                accuracy = Torch.asValue (Torch.mean matches) :: Double
            loss <- crossEntropy logits $ targets batch
            evaluate $ force (Torch.asValue loss :: Double, accuracy)
        }
  where
    weightsFile = "weights-" ++ task ++ ".txt"
    initialize :: [Double] -> IO [Torch.Tensor]
    initialize = go (zip [0 ..] shapes)
      where
        go [] [] = pure []
        go ((index, shape) : rest) values = do
            let (current, remaining) = splitAt (product shape) values
                tensor = Torch.toType Torch.Double $ Torch.asTensor current
                native
                    | index `elem` transposed =
                        Torch.contiguous $ Torch.transpose2D $ Torch.reshape (reverse shape) tensor
                    | otherwise = Torch.reshape shape tensor
            _ <- evaluate $ sum $ tensorValues native
            (native :) <$> go rest remaining
        go _ _ = fail "Invalid layer layout"

    canonical index tensor
        | index `elem` transposed = tensorValues $ Torch.transpose2D tensor
        | otherwise = tensorValues tensor

crossEntropy :: Torch.Tensor -> Torch.Tensor -> IO Torch.Tensor
crossEntropy = cast2 Native.cross_entropy_loss_tt

tensorValues :: Torch.Tensor -> [Double]
tensorValues = Torch.asValue . Torch.reshape [-1]

dense :: Torch.Tensor -> Torch.Tensor -> Torch.Tensor -> Torch.Tensor
dense weight bias =
    Torch.linear $ Torch.Linear (Torch.IndependentTensor weight) (Torch.IndependentTensor bias)

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
