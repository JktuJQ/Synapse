module Main (main) where

import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Monad (foldM, unless, void, when, zipWithM_)
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

data Model = Model
    { parameters :: [Torch.Parameter]
    , detachedParameters :: [Torch.Tensor]
    }

data Batch = Batch
    { inputs :: Torch.Tensor
    , targets :: Torch.Tensor
    }

main :: IO ()
main = do
    args <- getArgs
    case args of
        [directory, count] | Just epochs <- readMaybe count -> do
            runIris experiment directory epochs
        _ -> fail "Usage: hasktorch-iris DATA_DIRECTORY EPOCHS"

experiment :: Experiment Model Batch
experiment =
    Experiment
        { framework = "hasktorch"
        , executionNotes =
            "CPU LibTorch; explicit gradients; in-place SGD; RTS-managed GC"
        , revision =
            "0.2.2.0@059790c64e8556d8dd65978a2c38e50af84e6776"
        , loadParameters = \directory -> do
            tokens <- words <$> readFile (directory </> "weights.txt")
            values <- maybe (fail "Invalid initial weights") pure $ traverse readMaybe tokens
            let shapes = [[16, 4], [16], [3, 16], [3]]
            unless (length values == sum (map product shapes)) $ fail "Incorrect parameter count"
            tensors <- initialize shapes values
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
            evaluate $ force $ tensorValues $ forward (detachedParameters model) $ inputs batch
        , trainBatch = \model batch -> do
            let logits = forward (map Torch.toDependent $ parameters model) $ inputs batch
            loss <- crossEntropy logits $ targets batch
            value <- evaluate (Torch.asValue loss :: Double)
            gradients <- evaluate $ Torch.gradWithOptions (Torch.GradOptions False False False) loss (parameters model)
            alpha <- newScalar_d (-0.1)
            zipWithM_
                (\tensor gradient -> void (cast3 tensor_add__ts tensor gradient alpha :: IO Torch.Tensor))
                (detachedParameters model)
                gradients
            pure (value, model)
        , validateBatch = \model batch -> do
            let logits = forward (detachedParameters model) $ inputs batch
                predictions = Torch.argmax (Torch.Dim 1) Torch.RemoveDim logits
                matches = Torch.toType Torch.Double $ Torch.eq predictions $ targets batch
                accuracy = Torch.asValue (Torch.mean matches) :: Double
            loss <- crossEntropy logits $ targets batch
            evaluate $ force (Torch.asValue loss :: Double, accuracy)
        }
  where
    initialize :: [[Int]] -> [Double] -> IO [Torch.Tensor]
    initialize [] [] = pure []
    initialize (shape : rest) values = do
        let (current, remaining) = splitAt (product shape) values
            tensor = Torch.toType Torch.Double $ Torch.asTensor current
            native =
                if length shape == 2
                    then Torch.contiguous $ Torch.transpose2D $ Torch.reshape (reverse shape) tensor
                    else Torch.reshape shape tensor
        _ <- evaluate $ sum $ tensorValues native
        (native :) <$> initialize rest remaining
    initialize _ _ = fail "Invalid layer layout"
    canonical index tensor
        | index == 2 || index == 0 = tensorValues $ Torch.transpose2D tensor
        | otherwise = tensorValues tensor
    forward [w1, b1, w2, b2] input = dense w2 b2 $ Torch.tanh $ dense w1 b1 input
    forward _ _ = error "Expected two parameterized layers"
    dense weight bias =
        Torch.linear $ Torch.Linear (Torch.IndependentTensor weight) (Torch.IndependentTensor bias)

crossEntropy :: Torch.Tensor -> Torch.Tensor -> IO Torch.Tensor
crossEntropy = cast2 Native.cross_entropy_loss_tt

tensorValues :: Torch.Tensor -> [Double]
tensorValues = Torch.asValue . Torch.reshape [-1]

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
    (finalParameters, losses) <-
        foldM
            ( \(parameters, history) _ -> do
                (loss, next) <- update parameters training
                pure (next, loss : history)
            )
            (firstParameters, [firstLoss])
            [2 .. epochs]
    trainingEnd <- seconds
    (testLoss, accuracy) <- validateBatch spec finalParameters testing
    putStrLn $
        JSON.encode $
            JSON.makeObj
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
