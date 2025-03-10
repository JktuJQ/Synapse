{- | Implements batching - technology that allows packing and processing multiple samples at once.
-}


{- @TypeFamilies@ are needed to instantiate @DType@.
-}

{-# LANGUAGE TypeFamilies #-}


module Synapse.NN.Batching
    ( -- * @Sample@ datatype
    
      Sample (Sample, sampleInput, sampleOutput)

      -- * @Dataset@ datatype

    , Dataset (Dataset, unDataset)

    , shuffleDataset

    , VecDataset
    , BatchedDataset
    , batchVectors
    ) where


import Synapse.Tensors (DType, Indexable(unsafeIndex))

import Synapse.Tensors.Vec (Vec, size)

import Synapse.Tensors.Mat (Mat)
import qualified Synapse.Tensors.Mat as M

import Control.Monad.ST (runST)


import System.Random (RandomGen, uniformR)

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV


-- | @Sample@ datatype represents known pair of inputs and outputs of function that is unknown.
data Sample f a = Sample
    { sampleInput  :: f a  -- ^ Sample input.
    , sampleOutput :: f a  -- ^ Sample output.
    } deriving (Eq, Show)

type instance DType (Sample f a) = a


-- | @Dataset@ newtype wraps vector of @Sample@s - it represents known information about unknown function.
newtype Dataset f a = Dataset 
    { unDataset :: V.Vector (Sample f a)  -- ^ Unwraps @Dataset@ newtype.
    } deriving (Eq, Show)

type instance DType (Dataset f a) = a


-- | Shuffles any @Dataset@ using Fisher-Yates algorithm.
shuffleDataset :: RandomGen g => Dataset f a -> g -> (Dataset f a, g)
shuffleDataset (Dataset dataset) gen
    | V.length dataset <= 1 = (Dataset dataset, gen)
    | otherwise             = runST $ do
        mutableVector <- V.thaw dataset
        gen' <- go mutableVector (V.length dataset - 1) gen
        shuffledVector <- V.unsafeFreeze mutableVector
        return (Dataset shuffledVector, gen')
  where
    go _ 0 seed = return seed
    go v lastIndex seed = let (swapIndex, seed') = uniformR (0, lastIndex) seed
                          in MV.swap v swapIndex lastIndex >> go v lastIndex seed'

-- | @VecDataset@ type alias represents @Dataset@s with samples of vector functions.
type VecDataset = Dataset Vec
-- | @BatchedDataset@ type alias represents @Dataset@s with samples of vector functions where multiple samples were batched together.
type BatchedDataset = Dataset Mat

-- | Batches @VecDataset@ by grouping a given amount of samples into batches.
batchVectors :: Int -> VecDataset a -> BatchedDataset a
batchVectors batchSize (Dataset dataset) = Dataset $ V.fromList $ map groupBatch $ split dataset
  where
    split vector
        | V.length vector <= batchSize = [vector]
        | otherwise                  = let (current, remainder) = V.splitAt batchSize vector
                                       in current : split remainder
    
    groupBatch vector = let (rows, inputCols) = (V.length vector, size $ sampleInput $ V.unsafeIndex vector 0)
                            group (r, c) = unsafeIndex ((if c < inputCols then sampleInput else sampleOutput) (V.unsafeIndex vector r)) (c `mod` inputCols)
                            fullBatch = M.generate (rows, inputCols + size (sampleOutput $ V.unsafeIndex vector 0)) group
                            (batchInput, batchOutput, _, _) = M.split fullBatch (rows, inputCols)
                        in Sample batchInput batchOutput
