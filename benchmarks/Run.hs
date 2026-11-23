module Run (Options (..), runIris) where

import Control.Monad (forM, forM_, unless)
import Data.List (intercalate, sort, stripPrefix)
import Data.Maybe (fromMaybe, listToMaybe)
import GHC.Clock (getMonotonicTimeNSec)
import GHC.Conc (getNumProcessors)
import System.Directory (createDirectoryIfMissing, makeAbsolute)
import System.Environment (getEnvironment)
import System.Exit (ExitCode (..))
import System.FilePath ((</>))
import System.IO (hPutStrLn, stderr)
import System.IO.Error (tryIOError)
import System.Info qualified as System
import System.Process (CreateProcess (env), proc, readCreateProcessWithExitCode, readProcess)
import Text.Printf (printf)

import Data.ByteString qualified as BS
import Fixture (sha256)
import Text.JSON qualified as JSON

data Options = Options
    { dataset :: FilePath
    , synapse :: FilePath
    , hasktorch :: FilePath
    , grenade :: Maybe FilePath
    , output :: FilePath
    , repeats :: Int
    , epochs :: Int
    , threads :: Int
    , accelerateWorkers :: Maybe Int
    , blasThreads :: Maybe Int
    , rtsCapabilities :: Maybe Int
    }

type Object = [(String, JSON.JSValue)]

runIris :: Options -> IO ()
runIris options = do
    let threadCount = threads options
        workers = fromMaybe threadCount (accelerateWorkers options)
        blas = fromMaybe threadCount (blasThreads options)
        capabilities = fromMaybe threadCount (rtsCapabilities options)
        threadSettings =
            map
                (fmap show)
                [ ("OMP_NUM_THREADS", threadCount)
                , ("MKL_NUM_THREADS", blas)
                , ("OPENBLAS_NUM_THREADS", blas)
                , ("ACCELERATE_LLVM_NATIVE_THREADS", workers)
                ]
        frameworks =
            [("synapse", synapse options), ("hasktorch", hasktorch options)]
                ++ [("grenade", binary) | Just binary <- [grenade options]]
    unless (all (> 0) [repeats options, epochs options, threadCount, workers, blas, capabilities]) $
        fail "Repeats, epochs and parallelism settings must be positive"
    createDirectoryIfMissing True (output options)
    directory <- makeAbsolute (dataset options)
    manifest <- decodeJSON =<< readFile (directory </> "manifest.json")
    inherited <- getEnvironment
    let environment = threadSettings ++ filter (\(key, _) -> key `notElem` map fst threadSettings) inherited
    commands <- forM frameworks $ \(framework, binary) -> do
        path <- makeAbsolute binary
        pure (framework, path : [directory, show (epochs options), "+RTS", "-N" ++ show capabilities, "-RTS"])
    trials <- forM [0 .. repeats options - 1] $ \trial -> do
        let (before, after) = splitAt (trial `mod` length commands) commands
        forM (after ++ before) $ \(framework, command) -> do
            hPutStrLn stderr $ printf "Trial %d/%d: %s" (trial + 1) (repeats options) framework
            result <- runCommand environment command
            writeJSON (output options </> (framework ++ "-" ++ show (trial + 1) ++ ".json")) (object result)
            validate framework (epochs options) result
            pure (framework, result)
    checked <- traverse compareTrial trials
    summaries <- forM frameworks $ \(framework, _) -> do
        results <- traverse (lookupField framework) trials
        metrics <- forM timingFields $ \field -> do
            values <- traverse (timing field) results
            pure (field, values)
        pure (framework, metrics)
    revision <- trim <$> readProcess "git" ["rev-parse", "HEAD"] ""
    dirty <- not . null <$> readProcess "git" ["status", "--porcelain"] ""
    processors <- getNumProcessors
    cpuFile <- tryIOError (BS.readFile "/proc/cpuinfo")
    let processor = fromMaybe System.arch $ do
            bytes <- either (const Nothing) Just cpuFile
            label <-
                listToMaybe
                    [ value
                    | line <- lines (map (toEnum . fromIntegral) $ BS.unpack bytes)
                    , Just value <- [stripPrefix "model name" line]
                    ]
            pure $ dropWhile (\c -> c == ':' || c == '\t' || c == ' ') label
    fixtureHashes <- hashes [(name, directory </> name) | name <- ["train.csv", "test.csv", "weights.txt"]]
    binaryHashes <- hashes frameworks
    synapseHash <- lookupField "synapse" binaryHashes
    let summary =
            object
                [ (framework, object [(field, summarize values) | (field, values) <- metrics])
                | (framework, metrics) <- summaries
                ]
    writeJSON (output options </> "report.json") $
        object
            [ ("task", jsonString "iris-mlp")
            , ("reference_framework", jsonString "hasktorch")
            , ("revision", jsonString revision)
            , ("dirty", JSON.showJSON dirty)
            , ("platform", jsonString (System.os ++ "-" ++ System.arch))
            , ("processor", jsonString processor)
            , ("cpu_count", JSON.showJSON processors)
            , ("epochs", JSON.showJSON $ epochs options)
            , ("threads", JSON.showJSON threadCount)
            , ("rts_capabilities", JSON.showJSON capabilities)
            , ("commands", object (map (fmap JSON.showJSON) commands))
            , ("dataset", manifest)
            , ("thread_environment", object (map (fmap jsonString) threadSettings))
            , ("fixture_sha256", object fixtureHashes)
            , ("synapse_binary_sha256", synapseHash)
            , ("executable_sha256", object binaryHashes)
            , ("summary", summary)
            , ("trials", JSON.showJSON checked)
            ]
    putStrLn "framework\tfirst step (s)\ttraining (s)\ttraining after first step (s)\tprocess (s)"
    forM_ summaries $ \(framework, metrics) ->
        putStrLn $ intercalate "\t" (framework : [printf "%.6f" (median values) | (_, values) <- drop 1 metrics])

runCommand :: [(String, String)] -> [String] -> IO Object
runCommand environment (binary : arguments) = do
    start <- getMonotonicTimeNSec
    (status, stdout, errors) <- readCreateProcessWithExitCode (proc binary arguments){env = Just environment} ""
    end <- getMonotonicTimeNSec
    unless (status == ExitSuccess) $ fail $ unlines [show (binary : arguments), stdout, errors]
    fields <- JSON.fromJSObject <$> decodeJSON stdout
    let elapsed = fromIntegral (end - start) / 1e9 :: Double
    pure $ fields ++ [("process_seconds", JSON.showJSON elapsed), ("stderr", jsonString errors)]
runCommand _ [] = fail "Empty benchmark command"

validate :: String -> Int -> Object -> IO ()
validate framework count result = do
    actual <- getField "framework" result
    device <- getField "device" result
    dtype <- getField "dtype" result
    actualEpochs <- getField "epochs" result
    unless ((actual, device, dtype, actualEpochs) == (framework, "cpu", "float64", count)) $
        fail "Unexpected framework, device, dtype or epoch count"
    forM_ [("losses", count), ("first_step_parameters", 131), ("initial_logits", 360)] $ \(field, size) -> do
        values <- getField field result :: IO [Double]
        unless (length values == size && all finite values) $ fail $ "Invalid " ++ field
    forM_ (filter (/= "remaining_training_seconds") timingFields) $ \field -> do
        value <- getField field result
        unless (finite value && value >= 0) $ fail $ "Invalid " ++ field
    first <- getField "first_step_seconds" result
    total <- getField "training_seconds" result :: IO Double
    unless (first <= total) $ fail "First step exceeds total training time"
    loss <- getField "test_loss" result
    accuracy <- getField "test_accuracy" result
    unless (finite loss && loss >= 0) $ fail "Invalid test loss"
    unless (finite accuracy && accuracy >= 0 && accuracy <= 1) $ fail "Invalid accuracy"

compareTrial :: [(String, Object)] -> IO JSON.JSValue
compareTrial results = do
    reference <- lookupField "hasktorch" results
    errors <- forM [(name, fields) | (name, fields) <- results, name /= "hasktorch"] $ \(name, fields) -> do
        let check label a b =
                unless (close a b) $ fail $ name ++ " vs hasktorch: numerical comparison failed for " ++ label
        differences <- forM ["initial_logits", "first_step_parameters", "losses"] $ \field -> do
            left <- getField field fields
            right <- getField field reference
            unless (not (null left) && length left == length right) $ fail $ "Different or empty " ++ field
            let pairs = zip left right
            mapM_ (uncurry $ check field) pairs
            pure (field, JSON.showJSON $ maximum [abs (a - b) | (a, b) <- pairs])
        forM_ ["test_loss", "test_accuracy"] $ \field -> do
            left <- getField field fields
            right <- getField field reference
            check field left right
        pure (name, object differences)
    enriched <- forM results $ \(name, fields) -> do
        first <- getField "first_step_seconds" fields
        total <- getField "training_seconds" fields :: IO Double
        pure (name, object $ fields ++ [("remaining_training_seconds", JSON.showJSON $ total - first)])
    synapseErrors <- lookupField "synapse" errors
    pure $
        object
            [ ("results", object enriched)
            , ("maximum_absolute_errors", synapseErrors)
            , ("errors_by_framework", object errors)
            ]

timing :: String -> Object -> IO Double
timing "remaining_training_seconds" fields = (-) <$> getField "training_seconds" fields <*> getField "first_step_seconds" fields
timing field fields = getField field fields

timingFields :: [String]
timingFields =
    ["setup_seconds", "first_step_seconds", "training_seconds", "remaining_training_seconds", "process_seconds"]

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

close :: Double -> Double -> Bool
close a b = finite a && finite b && abs (a - b) <= max 1e-9 (1e-7 * max (abs a) (abs b))

median :: [Double] -> Double
median values =
    let ordered = sort values
        (middle, remainder) = length ordered `quotRem` 2
     in if remainder == 0
            then (ordered !! (middle - 1) + ordered !! middle) / 2
            else ordered !! middle

summarize :: [Double] -> JSON.JSValue
summarize values =
    object $
        map
            (fmap JSON.showJSON)
            [("median", median values), ("minimum", minimum values), ("maximum", maximum values)]

hashes :: [(String, FilePath)] -> IO Object
hashes = traverse $ \(name, path) -> do
    bytes <- BS.readFile path
    pure (name, jsonString $ sha256 bytes)

object :: Object -> JSON.JSValue
object = JSON.makeObj

jsonString :: String -> JSON.JSValue
jsonString = JSON.showJSON

writeJSON :: FilePath -> JSON.JSValue -> IO ()
writeJSON path value = writeFile path (JSON.encode value ++ "\n")

decodeJSON :: JSON.JSON a => String -> IO a
decodeJSON = fromJSONResult . JSON.decode

fromJSONResult :: JSON.Result a -> IO a
fromJSONResult (JSON.Ok value) = pure value
fromJSONResult (JSON.Error details) = fail details

lookupField :: String -> [(String, a)] -> IO a
lookupField name = maybe (fail $ "Missing " ++ name) pure . lookup name

getField :: JSON.JSON a => String -> Object -> IO a
getField name fields = fromJSONResult . JSON.readJSON =<< lookupField name fields

trim :: String -> String
trim = takeWhile (/= '\n')
