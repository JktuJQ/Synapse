module Data (Batch, readDataset) where

import Control.Monad (unless)
import Data.Char (isSpace)
import Data.List (elemIndex, mapAccumL, transpose)
import System.Random (mkStdGen, uniformShuffleList)
import Text.Read (readMaybe)

import Synapse.Tensor qualified as T

type Batch n = (T.TensorData '[n, 4] Double, T.TensorData '[n] Int)

readDataset :: FilePath -> IO (Batch 120, Batch 30)
readDataset path = do
    (training, testing) <- readSamples path
    let normalize = map $ standardize training
    pure (toBatch $ normalize training, toBatch $ normalize testing)
  where
    toBatch samples =
        let (features, labels) = unzip samples
         in (T.fromListData $ concat features, T.fromListData labels)

type Sample = ([Double], Int)

readSamples :: FilePath -> IO ([Sample], [Sample])
readSamples path = do
    rows <- filter (not . all isSpace) . lines <$> readFile path
    samples <- traverse parseRow rows
    let classes = [[sample | sample@(_, label) <- samples, label == category] | category <- [0 .. 2]]
    unless (map length classes == [50, 50, 50]) $ fail "Expected 50 samples per Iris class"
    let (_, shuffled) = mapAccumL shuffle (mkStdGen 42) classes
        (training, testing) = unzip $ map (splitAt 40) shuffled
    pure (concat training, concat testing)
  where
    shuffle generator samples = let (shuffled, next) = uniformShuffleList samples generator in (next, shuffled)
    parseRow row = case words $ map (\c -> if c == ',' then ' ' else c) row of
        [a, b, c, d, label]
            | Just values <- traverse readMaybe [a, b, c, d]
            , all (\x -> not (isNaN x || isInfinite x)) values
            , Just category <- elemIndex label ["Iris-setosa", "Iris-versicolor", "Iris-virginica"] ->
                pure (values, category)
        _ -> fail $ "Invalid Iris row: " ++ row

standardize :: [Sample] -> Sample -> Sample
standardize training = \(features, label) -> (zipWith3 normalize means scales features, label)
  where
    columns = transpose $ map fst training
    mean values = sum values / fromIntegral (length values)
    means = map mean columns
    scales = zipWith (\center values -> sqrt $ mean $ map (\x -> (x - center) ^ (2 :: Int)) values) means columns
    normalize center scale x = (x - center) / if scale == 0 then 1 else scale
