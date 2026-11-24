module IrisFixture (prepareIris) where

import Control.Monad (forM_, unless)
import Data.Char (isSpace)
import Data.List (elemIndex, intercalate, mapAccumL, transpose)
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.Random (mkStdGen, uniformShuffleList)
import Text.Read (readMaybe)

import Checksum (sha256)
import Data.ByteString qualified as BS
import Data.ByteString.Char8 qualified as BSC
import Task (Task (Iris), initialWeights)
import Text.JSON qualified as JSON

parseRow :: String -> IO ([Double], Int)
parseRow row = case words $ map (\c -> if c == ',' then ' ' else c) row of
    [a, b, c, d, label]
        | Just features <- traverse readMaybe [a, b, c, d]
        , all (\x -> not (isNaN x || isInfinite x)) features
        , Just target <- elemIndex label ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] ->
            pure (features, target)
    _ -> fail $ "Invalid Iris row: " ++ row

prepareIris :: FilePath -> FilePath -> IO ()
prepareIris source destination = do
    raw <- BS.readFile source
    samples <- traverse parseRow $ filter (not . all isSpace) $ lines $ BSC.unpack raw
    let groups = [[sample | sample@(_, label) <- samples, label == category] | category <- [0 .. 2]]
    unless (map length groups == [50, 50, 50]) $ fail "Expected 50 samples per Iris class"
    let splitSeed = 42
        initializationSeed = 17
        (afterGroups, shuffledGroups) = mapAccumL shuffle (mkStdGen splitSeed) groups
        (trainingGroups, testingGroups) = unzip $ map (splitAt 40) shuffledGroups
        (training, afterTraining) = uniformShuffleList (concat trainingGroups) afterGroups
        (testing, _) = uniformShuffleList (concat testingGroups) afterTraining
        columns = transpose (map fst training)
        means = map mean columns
        scales = zipWith (\center -> sqrt . mean . map (\x -> (x - center) ^ (2 :: Int))) means columns
        normalize (features, label) =
            intercalate "," $
                zipWith3 (\center scale x -> show ((x - center) / scale)) means scales features ++ [show label]
    unless (all (> 0) scales) $ fail "Expected positive feature scales"
    createDirectoryIfMissing True destination
    forM_ [("train.csv", training), ("test.csv", testing)] $ \(name, split) ->
        writeFile (destination </> name) $ unlines $ map normalize split
    writeFile (destination </> "weights.txt") $ unlines $ map show $ initialWeights initializationSeed Iris
    writeFile (destination </> "manifest.json") $
        (++ "\n") $
            JSON.encode $
                JSON.makeObj
                    [ ("source", JSON.showJSON source)
                    , ("source_sha256", JSON.showJSON $ sha256 raw)
                    ,
                        ( "config"
                        , JSON.showJSON $
                            JSON.makeObj
                                [ ("split_seed", JSON.showJSON splitSeed)
                                , ("initialization_seed", JSON.showJSON initializationSeed)
                                , ("generator", JSON.showJSON ("System.Random.StdGen" :: String))
                                ]
                        )
                    , ("feature_means", JSON.showJSON means)
                    , ("feature_scales", JSON.showJSON scales)
                    ]
  where
    mean values = sum values / fromIntegral (length values)
    shuffle generator samples = let (shuffled, next) = uniformShuffleList samples generator in (next, shuffled)
