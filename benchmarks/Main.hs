module Main (main) where

import Fixture (prepareIris)
import Options.Applicative
import Run (Options (..), runIris)

data Command = Prepare FilePath FilePath | Compare Options

main :: IO ()
main = do
    parsedCommand <- execParser $ info (helper <*> commands) $ progDesc "Prepare and compare Iris benchmarks"
    case parsedCommand of
        Prepare source destination -> prepareIris source destination
        Compare options -> runIris options

commands :: Parser Command
commands =
    hsubparser $
        command "prepare" (info prepare $ progDesc "Prepare Iris data and initial weights")
            <> command "iris" (info (Compare <$> comparison) $ progDesc "Compare Iris implementations")

prepare :: Parser Command
prepare =
    Prepare
        <$> strArgument (metavar "IRIS_DATA")
        <*> strArgument (metavar "DESTINATION")

comparison :: Parser Options
comparison =
    Options
        <$> path "data" "FIXTURE_DIRECTORY"
        <*> path "synapse" "EXECUTABLE"
        <*> path "hasktorch" "EXECUTABLE"
        <*> optional (path "grenade" "EXECUTABLE")
        <*> path "output" "RESULTS_DIRECTORY"
        <*> count "repeats" 5
        <*> count "epochs" 200
        <*> count "threads" 1
        <*> optional (option auto $ long "accelerate-workers" <> metavar "N")
        <*> optional (option auto $ long "blas-threads" <> metavar "N")
        <*> optional (option auto $ long "rts-capabilities" <> metavar "N")
  where
    path name label = strOption (long name <> metavar label)
    count name fallback = option auto (long name <> metavar "N" <> value fallback <> showDefault)
