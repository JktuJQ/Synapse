{- | Implements batching - technology that allows packing and processing multiple samples at once.
-}


-- @TypeFamilies@ are needed to instantiate @DType@.
{-# LANGUAGE TypeFamilies #-}


module Synapse.NN.Batching
    ( -- * @Sample@ datatype
    
      Sample (Sample, sampleInput, sampleOutput)

      -- * @Dataset@ datatype

    , Dataset (Dataset, unDataset)

    , datasetSize

    , shuffleDataset
    , splitDataset

    , VecDataset
    , BatchedDataset
    , batchVectors
    ) where


import Synapse.Tensors (DType, Indexable(unsafeIndex))

import Synapse.Tensors.Vec (Vec(Vec))
import qualified Synapse.Tensors.Vec as V

import Synapse.Tensors.Mat (Mat)
import qualified Synapse.Tensors.Mat as M

import Control.Monad.ST (runST)

import System.Random (RandomGen, uniformR)

import Data.Vector (thaw, unsafeFreeze)
import Data.Vector.Mutable (swap)


-- | @Sample@ datatype represents known pair of inputs and outputs of function that is unknown.
data Sample a = Sample
    { sampleInput  :: a  -- ^ Sample input.
    , sampleOutput :: a  -- ^ Sample output.
    } deriving (Eq, Show)

type instance DType (Sample a) = DType a


-- | @Dataset@ newtype wraps vector of @Sample@s - it represents known information about unknown function.
newtype Dataset a = Dataset 
    { unDataset :: Vec (Sample a)  -- ^ Unwraps @Dataset@ newtype.
    } deriving (Eq, Show)

type instance DType (Dataset a) = DType a

-- | Returns size of dataset.
datasetSize :: Dataset a -> Int
datasetSize = V.size . unDataset


-- | Shuffles any @Dataset@ using Fisher-Yates algorithm.
shuffleDataset :: RandomGen g => Dataset a -> g -> (Dataset a, g)
shuffleDataset (Dataset dataset) gen
    | V.size dataset <= 1 = (Dataset dataset, gen)
    | otherwise           = runST $ do
        mutableVector <- thaw $ V.unVec dataset
        gen' <- go mutableVector (V.size dataset - 1) gen
        shuffledVector <- unsafeFreeze mutableVector
        return (Dataset $ Vec shuffledVector, gen')
  where
    go _ 0 seed = return seed
    go v lastIndex seed = let (swapIndex, seed') = uniformR (0, lastIndex) seed
                          in swap v swapIndex lastIndex >> go v (lastIndex - 1) seed'

-- | Splits dataset such that size of left dataset divided on size of right dataset will be equal to given ratio.
splitDataset :: Dataset a -> Float -> (Dataset a, Dataset a)
splitDataset (Dataset dataset) ratio = let (left, right) = V.splitAt (round $ fromIntegral (V.size dataset) * ratio) dataset
                                       in (Dataset left, Dataset right)


-- | @VecDataset@ type alias represents @Dataset@s with samples of vector functions.
type VecDataset a = Dataset (Vec a)
-- | @BatchedDataset@ type alias represents @Dataset@s with samples of vector functions where multiple samples were batched together.
type BatchedDataset a = Dataset (Mat a)

-- | Batches @VecDataset@ by grouping a given amount of samples into batches.
batchVectors :: Int -> VecDataset a -> BatchedDataset a
batchVectors batchSize (Dataset dataset) = Dataset $ V.fromList $ map groupBatch $ split dataset
  where
    split vector
        | V.size vector <= batchSize = [vector]
        | otherwise                  = let (current, remainder) = V.splitAt batchSize vector
                                       in current : split remainder
    
    groupBatch vector = let (rows, inputCols) = (V.size vector, V.size $ sampleInput $ unsafeIndex vector 0)
                            group (r, c) = unsafeIndex ((if c < inputCols then sampleInput else sampleOutput) (unsafeIndex vector r)) (c `mod` inputCols)
                            fullBatch = M.generate (rows, inputCols + V.size (sampleOutput $ unsafeIndex vector 0)) group
                            (batchInput, batchOutput, _, _) = M.split fullBatch (rows, inputCols)
                        in Sample batchInput batchOutput
