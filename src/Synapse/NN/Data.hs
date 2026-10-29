{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

-- | Load paired tensor examples on demand and form batches with a runtime-selected leading dimension.
-- Sources may read from memory or disk without loading the entire dataset in advance.
module Synapse.NN.Data
  ( Dataset (..),
    getSample,
    loadBatches,
    LastBatch (..),
    Batch (..),
    SomeBatch (..),
    shuffle,
    withBatch,
    batchSize,
    batchMask,
  )
where

import qualified Data.Array as Array
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Unboxed as Unboxed
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat, SomeNat (..), natVal, someNatVal)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shapeSize)
import Synapse.Tensor.Type (DType, TensorData (..))
import System.Random (RandomGen, uniformShuffleList)

-- | A finite collection of examples loaded by index, individually or in groups.
-- The size must be nonnegative, and valid indices range from zero to 'datasetSize' minus one.
-- Pure sources can use @Identity@ from "Data.Functor.Identity"; disk sources typically use @IO@.
data Dataset m sample
  = Indexed
      { -- | Number of examples available without loading them.
        datasetSize :: Int,
        -- | Read one example, decoding or transforming it as needed.
        readSample :: Int -> m sample
      }
  | Grouped
      { datasetSize :: Int,
        -- | Read a group of examples in the requested order, preserving repeated indices.
        -- Return exactly one example for each supplied index.
        -- The loader assembles these examples into a batch afterwards.
        readSamples :: [Int] -> m [sample]
      }

-- | Obtain examples using the source's individual or grouped reader.
fetchSamples :: (Applicative m) => Dataset m sample -> [Int] -> m [sample]
fetchSamples (Indexed _ reader) indices = traverse reader indices
fetchSamples (Grouped _ reader) indices = checkCount <$> reader indices
  where
    sameCount [] [] = True
    sameCount (_ : requested) (_ : samples) = sameCount requested samples
    sameCount _ _ = False
    checkCount samples
      | sameCount indices samples = samples
      | otherwise = error "Synapse.NN.loadBatches: grouped reader must return one example per requested index"

-- | Read one example from either source type.
-- A grouped source receives a singleton list of indices.
getSample :: (Applicative m) => Dataset m sample -> Int -> m sample
getSample source index = extract <$> fetchSamples source [index]
  where
    extract [sample] = sample
    extract _ = error "Synapse.NN.getSample: expected one example"

-- | Treatment of a final batch smaller than the requested size.
data LastBatch features targets
  = -- | Retain the remaining examples with their actual batch size.
    KeepLast
  | -- | Discard the remaining examples without loading them.
    DropLast
  | -- | Fill missing examples with the supplied input and target element values.
    -- All elements of each added example receive the corresponding fill value.
    PadLast features targets
  deriving (Eq, Show)

-- | Paired batches with a shared leading dimension.
-- 'loadBatches' produces batches containing at least one real example.
data Batch batch in_sh out_sh features targets = Batch
  { -- | Input examples, including any padding, along the leading axis.
    batchInputs :: TensorData (batch ': in_sh) features,
    -- | Target examples in the same order as the inputs, including any padding.
    batchTargets :: TensorData (batch ': out_sh) targets,
    -- | Number of real examples, occupying the first positions of the leading axis.
    -- When constructing a batch, this must be between zero and its 'batchSize'.
    validCount :: Int
  }

-- | A batch whose leading dimension is determined at runtime.
-- 'withBatch' makes the dimension and its shape evidence available to a polymorphic handler.
data SomeBatch in_sh out_sh features targets
  = forall batch.
    (KnownNat batch, Shape (batch ': in_sh), Shape (batch ': out_sh)) =>
    SomeBatch (Batch batch in_sh out_sh features targets)

-- | Number of examples along the leading axis, including padding.
batchSize :: forall batch in_sh out_sh features targets. (KnownNat batch) => Batch batch in_sh out_sh features targets -> Int
batchSize _ = fromIntegral (natVal (Proxy @batch))

-- | Fill host storage by linear index, then attach the requested tensor shape without copying its contents.
generateData :: forall sh element. (Shape sh, DType element) => (Int -> element) -> TensorData sh element
generateData element =
  let values = A.fromFunction (A.Z A.:. shapeSize @sh) (\(A.Z A.:. index) -> element index)
   in TensorData (A.arrayReshape (backendShape (Proxy @sh)) values)

-- | Uniformly permute source indices without loading any examples.
-- Returns a reordered source and the next random generator.
-- Only an index permutation occupies additional memory proportional to 'datasetSize'.
-- A grouped source still receives one request per batch, with all indices remapped together.
-- For a new epoch, shuffle the original source with the returned generator.
shuffle :: (RandomGen g) => Dataset m sample -> g -> (Dataset m sample, g)
shuffle dataset generator =
  let count = datasetSize dataset
      (indices, next) = uniformShuffleList [0 .. count - 1] generator
      order :: Unboxed.UArray Int Int
      order = Unboxed.listArray (0, count - 1) indices
      reordered = case dataset of
        Indexed _ reader -> Indexed count (reader . (order Unboxed.!))
        Grouped _ reader -> Grouped count (reader . map (order Unboxed.!))
   in (reordered, next)

-- | One boolean per example, with 'True' for real examples and 'False' for padding.
-- The mask is constructed on demand and does not automatically affect losses or other computations.
-- Broadcast it explicitly when selecting values with additional feature axes.
batchMask :: forall batch in_sh out_sh features targets. (KnownNat batch) => Batch batch in_sh out_sh features targets -> TensorData '[batch] Bool
batchMask batch = generateData @'[batch] (< validCount batch)

-- | Process a batch with its runtime-selected dimension available at the type level.
-- The result type must not depend on the hidden batch dimension.
withBatch :: SomeBatch in_sh out_sh features targets -> (forall batch. (KnownNat batch, Shape (batch ': in_sh), Shape (batch ': out_sh)) => Batch batch in_sh out_sh features targets -> r) -> r
withBatch (SomeBatch batch) handler = handler batch

-- | Create loading actions for one pass over a source in its current order.
-- Each action obtains only its own batch; constructing or traversing the action list does not load examples.
-- Discarded final examples are not read when using 'DropLast'.
-- Execute and consume batches one at a time to avoid retaining an entire epoch in memory.
-- Indexed sources read each example separately; grouped sources receive one request per batch.
-- Returned examples are copied into the destination arrays without converting tensor elements to lists.
--
-- @
-- let (epoch, nextGenerator) = shuffle dataset generator
-- forM_ (loadBatches 32 KeepLast epoch) $ \load -> do
--   someBatch <- load
--   withBatch someBatch $ \batch -> ...
-- @
--
-- For pure sources, @map runIdentity (loadBatches 32 KeepLast dataset)@ yields a lazy list of batches.
loadBatches ::
  forall m in_sh out_sh features targets.
  ( Applicative m,
    Shape in_sh,
    Shape out_sh,
    forall batch. (KnownNat batch) => Shape (batch ': in_sh),
    forall batch. (KnownNat batch) => Shape (batch ': out_sh),
    DType features,
    DType targets
  ) =>
  Int ->
  LastBatch features targets ->
  Dataset m (TensorData in_sh features, TensorData out_sh targets) ->
  [m (SomeBatch in_sh out_sh features targets)]
loadBatches limit policy dataset
  | limit <= 0 = error "Synapse.NN.loadBatches: requested size must be positive"
  | otherwise = emit 0 (datasetSize dataset)
  where
    copyExamples :: forall batch sh element. (Shape sh, Shape (batch ': sh), DType element) => Int -> Maybe element -> (Int -> Int -> element) -> TensorData (batch ': sh) element
    copyExamples valid padding readElement = generateData @(batch ': sh) $ \index ->
      let (row, offset) = index `quotRem` shapeSize @sh
       in if row < valid
            then readElement row offset
            else case padding of
              Just value -> value
              Nothing -> error "Synapse.NN.loadBatches: missing padding value"

    loadChunk valid indices =
      let (size, inputPadding, targetPadding) = case policy of
            PadLast x y -> (limit, Just x, Just y)
            _ -> (valid, Nothing, Nothing)
       in case someNatVal (fromIntegral size) of
            SomeNat (_ :: Proxy batch) ->
              let collate samples =
                    let table = Array.listArray (0, valid - 1) samples
                        inputs row = A.linearIndexArray (unTensorData (fst (table Array.! row)))
                        targets row = A.linearIndexArray (unTensorData (snd (table Array.! row)))
                     in SomeBatch (Batch (copyExamples @batch @in_sh valid inputPadding inputs) (copyExamples @batch @out_sh valid targetPadding targets) valid)
               in collate <$> fetchSamples dataset indices

    emit offset remaining
      | remaining <= 0 = []
      | DropLast <- policy, remaining < limit = []
      | otherwise =
          let valid = min limit remaining
           in loadChunk valid [offset .. offset + valid - 1] : emit (offset + valid) (remaining - valid)
