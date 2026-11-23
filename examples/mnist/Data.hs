module Data (readDataset) where

import Control.Monad (unless)
import Data.Functor.Identity (Identity)
import Data.Word (Word32)
import System.FilePath ((</>))

import Data.ByteString qualified as BS
import Synapse.NN qualified as NN
import Synapse.Tensor qualified as T

type Sample = (T.TensorData '[1, 28, 28] Double, T.TensorData '[] Int)

readDataset :: FilePath -> String -> IO (NN.Dataset Identity Sample)
readDataset directory split = do
    images <- BS.readFile $ directory </> split ++ "-images-idx3-ubyte"
    labels <- BS.readFile $ directory </> split ++ "-labels-idx1-ubyte"
    unless (BS.length images >= 16 && BS.length labels >= 8) $ fail "Truncated MNIST headers"
    let count = fromIntegral $ word32 labels 4
    unless
        ( word32 images 0 == 2051
            && word32 labels 0 == 2049
            && word32 images 4 == fromIntegral count
            && word32 images 8 == 28
            && word32 images 12 == 28
            && count > 0
            && BS.length images == 16 + count * 784
            && BS.length labels == 8 + count
            && BS.all (< 10) (BS.drop 8 labels)
        )
        $ fail "Expected matching MNIST images and labels"

    pure $ NN.Indexed count $ \index ->
        let pixels = BS.take 784 $ BS.drop (16 + index * 784) images
            label = fromIntegral $ BS.index labels (8 + index)
         in pure (T.fromListData $ map ((/ 255) . fromIntegral) $ BS.unpack pixels, T.fromListData [label])

word32 :: BS.ByteString -> Int -> Word32
word32 bytes offset =
    BS.foldl' (\value byte -> value * 256 + fromIntegral byte) 0 $ BS.take 4 $ BS.drop offset bytes
