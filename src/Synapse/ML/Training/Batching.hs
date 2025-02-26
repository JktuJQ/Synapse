{- | Implements batching - technology that allows packing and processing multiple samples at once.
-}


module Synapse.ML.Training.Batching
    ( -- * @Sample@ datatype
    
      Sample (Sample, sampleInput, sampleOutput)

      -- * @Dataset@ datatype

    , Dataset (Dataset, unDataset)

    , shuffleDataset

    , VectorDataset
    , BatchedDataset
    , batchVectors
    ) where


import Synapse.LinearAlgebra (Indexable(unsafeIndex))

import Synapse.LinearAlgebra.Vec (Vec (Vec))
import qualified Synapse.LinearAlgebra.Vec as V

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M

import Control.Monad.ST (runST)

import Data.Vector (thaw, unsafeFreeze)
import Data.Vector.Mutable (swap)

import System.Random (RandomGen, uniformR)


-- | @Sample@ datatype represents known pair of inputs and outputs of function that is unknown.
data Sample f a = Sample
    { sampleInput  :: f a  -- ^ Sample input.
    , sampleOutput :: f a  -- ^ Sample output.
    }


-- | @Dataset@ newtype wraps @Vec@ of @Sample@s - it represents known information about unknown function.
newtype Dataset f a = Dataset 
    { unDataset :: Vec (Sample f a)  -- ^ Unwraps @Dataset@ newtype.
    }

-- | Shuffles any @Dataset@ using Fisher-Yates algorithm.
shuffleDataset :: RandomGen g => Dataset f a -> g -> (Dataset f a, g)
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
                          in swap v swapIndex lastIndex >> go v lastIndex seed'

-- | @VectorDataset@ type alias represents @Dataset@s with samples of vector functions.
type VectorDataset = Dataset Vec
-- | @BatchedDataset@ type alias represents @Dataset@s with samples of vector functions where multiple samples were batched together.
type BatchedDataset = Dataset Mat

-- | Batches @VectorDataset@ by grouping a given amount of samples into batches.
batchVectors :: Int -> VectorDataset a -> BatchedDataset a
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
