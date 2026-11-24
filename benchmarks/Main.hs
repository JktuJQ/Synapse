module Main (main) where

import IrisFixture (prepareIris)
import MnistFixture (prepareMnist)
import Options.Applicative
import Run (Options (..), runComparison)

import Task (Task (..), taskName)

data Command
    = PrepareIris FilePath FilePath
    | PrepareMnist FilePath FilePath Int Int
    | Compare Task Options

count :: String -> Int -> Parser Int
count name fallback = option auto (long name <> metavar "N" <> value fallback <> showDefault)

irisPreparation :: Parser Command
irisPreparation =
    PrepareIris
        <$> strArgument (metavar "IRIS_DATA")
        <*> strArgument (metavar "DESTINATION")

mnistPreparation :: Parser Command
mnistPreparation =
    PrepareMnist
        <$> strArgument (metavar "MNIST_DIRECTORY")
        <*> strArgument (metavar "DESTINATION")
        <*> count "train-count" 60000
        <*> count "test-count" 10000

comparison :: Int -> Parser Options
comparison defaultEpochs =
    Options
        <$> path "data" "FIXTURE_DIRECTORY"
        <*> path "synapse" "EXECUTABLE"
        <*> path "hasktorch" "EXECUTABLE"
        <*> optional (path "grenade" "EXECUTABLE")
        <*> path "output" "RESULTS_DIRECTORY"
        <*> count "repeats" 5
        <*> count "epochs" defaultEpochs
        <*> count "threads" 1
        <*> optional (option auto $ long "accelerate-workers" <> metavar "N")
        <*> optional (option auto $ long "blas-threads" <> metavar "N")
        <*> optional (option auto $ long "rts-capabilities" <> metavar "N")
  where
    path name label = strOption (long name <> metavar label)

commands :: Parser Command
commands =
    hsubparser $
        command "prepare-iris" (info irisPreparation $ progDesc "Prepare Iris data and initial weights")
            <> command "prepare-mnist" (info mnistPreparation $ progDesc "Prepare MNIST IDX data and initial weights")
            <> command "iris" (info (Compare Iris <$> comparison 200) $ progDesc "Compare Iris implementations")
            <> foldMap mnistCommand [minBound .. maxBound]
  where
    mnistCommand model =
        let task = Mnist model
         in command (taskName task) (info (Compare task <$> comparison 3) $ progDesc ("Compare " ++ taskName task))

main :: IO ()
main = do
    parsedCommand <- execParser $ info (helper <*> commands) $ progDesc "Prepare and compare framework benchmarks"
    case parsedCommand of
        PrepareIris source destination -> prepareIris source destination
        PrepareMnist source destination trainCount testCount -> prepareMnist source destination trainCount testCount
        Compare task options -> runComparison task options
