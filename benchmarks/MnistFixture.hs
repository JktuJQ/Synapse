module MnistFixture (prepareMnist) where

import Control.Monad (forM_, unless)
import Data.Bits (shiftR)
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.Random (mkStdGen, uniformShuffleList)

import Checksum (sha256)
import Data.ByteString qualified as BS
import Task (Task (Mnist), initialWeights, modelName)
import Text.JSON qualified as JSON

word32 :: BS.ByteString -> Int -> Int
word32 bytes offset = BS.foldl' (\value byte -> value * 256 + fromIntegral byte) 0 $ BS.take 4 $ BS.drop offset bytes

prepareSplit :: FilePath -> FilePath -> Int -> (String, String, Int, Int) -> IO [(String, JSON.JSValue)]
prepareSplit source destination seed (input, output, count, limit) = do
    let imageName = input ++ "-images-idx3-ubyte"
        labelName = input ++ "-labels-idx1-ubyte"
    images <- BS.readFile (source </> imageName)
    labels <- BS.readFile (source </> labelName)
    let available = word32 labels 4
    unless
        ( word32 images 0 == 2051
            && word32 labels 0 == 2049
            && word32 images 4 == available
            && word32 images 8 == 28
            && word32 images 12 == 28
            && available <= limit
            && available >= count
            && BS.length images == 16 + available * 784
            && BS.length labels == 8 + available
            && BS.all (< 10) (BS.drop 8 labels)
        )
        $ fail
        $ "Invalid or insufficient MNIST IDX data in " ++ input
    let order =
            take count $
                if output == "train"
                    then fst (uniformShuffleList [0 .. available - 1] (mkStdGen seed))
                    else [0 .. available - 1]
        countBytes = BS.pack [fromIntegral (count `shiftR` shift) | shift <- [24, 16, 8, 0]]
        payload = BS.concat [BS.take 784 (BS.drop (16 + 784 * index) images) | index <- order]
        classes = BS.pack [BS.index labels (8 + index) | index <- order]
    BS.writeFile (destination </> (output ++ "-images.idx")) $
        BS.take 4 images <> countBytes <> BS.take 8 (BS.drop 8 images) <> payload
    BS.writeFile (destination </> (output ++ "-labels.idx")) $
        BS.take 4 labels <> countBytes <> classes
    pure [(imageName, JSON.showJSON $ sha256 images), (labelName, JSON.showJSON $ sha256 labels)]

prepareMnist :: FilePath -> FilePath -> Int -> Int -> IO ()
prepareMnist source destination trainCount testCount = do
    let shuffleSeed = 42
        initializationSeed = 17
        splits = [("train", "train", trainCount, 60000), ("t10k", "test", testCount, 10000)]
    forM_ splits $ \(_, _, count, limit) ->
        unless (count > 0 && count <= limit && count `mod` 100 == 0) $
            fail $
                "Expected a positive multiple of 100 up to " ++ show limit
    createDirectoryIfMissing True destination
    sourceHashes <- traverse (prepareSplit source destination shuffleSeed) splits
    forM_ [minBound .. maxBound] $ \model ->
        writeFile (destination </> ("weights-" ++ modelName model ++ ".txt")) $
            unlines $
                map show $
                    initialWeights initializationSeed (Mnist model)
    writeFile (destination </> "manifest.json") $
        (++ "\n") $
            JSON.encode $
                JSON.makeObj
                    [ ("dataset", JSON.showJSON ("mnist" :: String))
                    , ("train_count", JSON.showJSON trainCount)
                    , ("test_count", JSON.showJSON testCount)
                    , ("batch_size", JSON.showJSON (100 :: Int))
                    , ("shuffle_seed", JSON.showJSON shuffleSeed)
                    , ("initialization_seed", JSON.showJSON initializationSeed)
                    , ("source_sha256", JSON.makeObj $ concat sourceHashes)
                    ]
