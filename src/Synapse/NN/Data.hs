{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE TypeFamilies #-}

{- | Load examples by index, shuffle their order, and assemble input and target tensors into minibatches.
A 'Dataset' supplies examples in a context chosen by the caller, such as @Identity@ or @IO@.
'loadBatches' returns loading actions; execute each action when its batch is needed.

'LastBatch' selects whether to keep, discard, or pad an incomplete final batch.
Each 'SomeBatch' hides its leading dimension until 'withBatch' introduces it into scope.
Padding is recorded by 'validCount' and 'batchMask'; losses must exclude padded examples explicitly.

A 'BatchPlan' additionally records the expected batch sizes, without loading examples.
'prepareBatchStep' uses this information to prepare a computation for each size.
-}
module Synapse.NN.Data
    ( -- * Sources
      Dataset (..)
    , fetchSample
    , shuffle

      -- * Batches
    , LastBatch (..)
    , Batch (..)
    , SomeBatch (..)
    , withBatch
    , batchSize
    , batchMask
    , loadBatches

      -- * Loading plans
    , BatchPlan
    , batchPlan
    , planBatches
    , batchSizes
    , batchLoads

      -- * Prepared computations
    , BatchStep (..)
    , UnexpectedSize (..)
    , prepareBatchStep
    )
where

import Control.Monad.Trans.State.Strict (State, state)
import Data.Proxy (Proxy (..))
import Data.Type.Equality ((:~:) (Refl))
import GHC.TypeNats (KnownNat, SomeNat (..), natVal, sameNat, someNatVal)
import System.Random (RandomGen, uniformShuffleList)

import Synapse.Tensor.Backend (Backend, RunnableTensor, run)
import Synapse.Tensor.Representation (DType, Tensor, TensorData (..))
import Synapse.Tensor.Shapes.Backend (backendShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Synapse.Tensor.Structure (WithTensor)

import Data.Array qualified as Array
import Data.Array.Accelerate qualified as A
import Data.Array.Unboxed qualified as Unboxed
import Data.Map.Lazy qualified as Map
import Data.Set qualified as Set

-- | A prepared function paired with evidence of its batch dimension.
data Prepared args in_sh out_sh features targets result
    = forall batch. KnownNat batch => Prepared
        (Proxy batch)
        (args -> Batch batch in_sh out_sh features targets -> result)

-- | Fill host storage by linear index, then attach the requested tensor shape without copying its contents.
generateData
    :: forall sh element
     . (Shape sh, DType element)
    => (Int -> element)
    -> TensorData sh element
generateData element =
    let values = A.fromFunction (A.Z A.:. shapeSize @sh) (\(A.Z A.:. index) -> element index)
     in TensorData (A.arrayReshape (backendShape (Proxy @sh)) values)

{- | A finite collection of examples loaded by index, individually or in groups.
-
The size must be nonnegative, and valid indices range from zero to 'datasetSize' minus one.
Readers must return the requested examples in order, including repeated indices.
A grouped reader must return exactly one example per requested index.
Pure sources can use @Identity@ from "Data.Functor.Identity"; disk sources typically use @IO@.
-}
data Dataset m sample
    = Indexed
        { datasetSize :: Int
        , readSample :: Int -> m sample
        }
    | Grouped
        { datasetSize :: Int
        , readSamples :: [Int] -> m [sample]
        }

-- | Read one example from either source type.
fetchSample
    :: Applicative m
    => Dataset m sample
    -> Int
    -> m sample
fetchSample source index = extract <$> fetchSamples source [index]
  where
    extract [sample] = sample
    extract _ = error "Synapse.NN.fetchSample: expected one example"

-- | Read multiple examples from either source type.
fetchSamples
    :: Applicative m
    => Dataset m sample
    -> [Int]
    -> m [sample]
fetchSamples (Indexed _ reader) indices = traverse reader indices
fetchSamples (Grouped _ reader) indices = checkCount <$> reader indices
  where
    sameCount [] [] = True
    sameCount (_ : requested) (_ : samples) = sameCount requested samples
    sameCount _ _ = False
    checkCount samples
        | sameCount indices samples = samples
        | otherwise =
            error "Synapse.NN.loadBatches: grouped reader must return one example per requested index"

{- | Uniformly permute source indices without loading any examples.
Only an index permutation occupies additional memory proportional to 'datasetSize'.
-}
shuffle
    :: RandomGen g
    => Dataset m sample
    -> State g (Dataset m sample)
shuffle dataset = state $ \generator ->
    let count = datasetSize dataset
        (indices, next) = uniformShuffleList [0 .. count - 1] generator
        order = Unboxed.listArray @Unboxed.UArray (0, count - 1) indices
        reordered = case dataset of
            Indexed _ reader -> Indexed count (reader . (order Unboxed.!))
            Grouped _ reader -> Grouped count (reader . map (order Unboxed.!))
     in (reordered, next)

-- | Treatment of a final batch smaller than the requested size.
data LastBatch features targets
    = KeepLast
    | DropLast
    | PadLast features targets
    deriving (Eq, Show)

-- | Paired batches with a shared leading dimension.
data Batch batch in_sh out_sh features targets = Batch
    { batchInputs :: TensorData (batch ': in_sh) features
    , batchTargets :: TensorData (batch ': out_sh) targets
    , validCount :: Int
    }

{- | A batch whose leading dimension is determined at runtime.
'withBatch' makes the dimension and its shape evidence available to a polymorphic handler.
-}
data SomeBatch in_sh out_sh features targets
    = forall batch.
        (KnownNat batch, Shape (batch ': in_sh), Shape (batch ': out_sh)) =>
      SomeBatch (Batch batch in_sh out_sh features targets)

-- | Process a batch with its runtime-selected dimension available at the type level.
withBatch
    :: SomeBatch in_sh out_sh features targets
    -> ( forall batch
          . (KnownNat batch, Shape (batch ': in_sh), Shape (batch ': out_sh))
         => Batch batch in_sh out_sh features targets
         -> r
       )
    -> r
withBatch (SomeBatch batch) handler = handler batch

-- | Number of examples along the leading axis, including padding.
batchSize
    :: forall batch in_sh out_sh features targets
     . KnownNat batch
    => Batch batch in_sh out_sh features targets
    -> Int
batchSize _ = fromIntegral (natVal (Proxy @batch))

{- | One boolean per example, with 'True' for real examples and 'False' for padding.
-
The mask is constructed on demand and does not automatically affect losses or other computations.
Broadcast it explicitly when selecting values with additional feature axes.
-}
batchMask
    :: forall batch in_sh out_sh features targets
     . KnownNat batch
    => Batch batch in_sh out_sh features targets
    -> TensorData '[batch] Bool
batchMask batch = generateData @'[batch] (< validCount batch)

{- | Create loading actions for one pass over a source in its current order.
-
Each action obtains only its own batch; constructing or traversing the action list does not load examples.
Discarded final examples are not read when using 'DropLast'.
Execute and consume batches one at a time to avoid retaining an entire epoch in memory.
The requested size must be positive, and the dataset size must be nonnegative.
An empty source produces no actions.

@
let (epoch, nextGenerator) = runState (shuffle dataset) generator
forM_ (loadBatches 32 KeepLast epoch) $ \\load -> do
  someBatch <- load
  withBatch someBatch $ \\batch -> ...
@

For pure sources, @map runIdentity (loadBatches 32 KeepLast dataset)@ yields a lazy list of batches.
-}
loadBatches
    :: forall m in_sh out_sh features targets
     . ( Applicative m
       , Shape in_sh
       , Shape out_sh
       , forall batch. KnownNat batch => Shape (batch ': in_sh)
       , forall batch. KnownNat batch => Shape (batch ': out_sh)
       , DType features
       , DType targets
       )
    => Int
    -> LastBatch features targets
    -> Dataset m (TensorData in_sh features, TensorData out_sh targets)
    -> [m (SomeBatch in_sh out_sh features targets)]
loadBatches limit policy dataset
    | limit <= 0 = error "Synapse.NN.loadBatches: requested size must be positive"
    | otherwise = emit 0 (datasetSize dataset)
  where
    copyExamples
        :: forall batch sh element
         . (Shape sh, Shape (batch ': sh), DType element)
        => Int
        -> Maybe element
        -> (Int -> Int -> element)
        -> TensorData (batch ': sh) element
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
                             in SomeBatch
                                    ( Batch
                                        (copyExamples @batch @in_sh valid inputPadding inputs)
                                        (copyExamples @batch @out_sh valid targetPadding targets)
                                        valid
                                    )
                     in collate <$> fetchSamples dataset indices

    -- Produce one loading action per retained chunk.
    emit offset remaining
        | remaining <= 0 = []
        | DropLast <- policy, remaining < limit = []
        | otherwise =
            let valid = min limit remaining
             in loadChunk valid [offset .. offset + valid - 1] : emit (offset + valid) (remaining - valid)

-- | Expected leading dimensions and lazy loading actions for a source.
data BatchPlan m in_sh out_sh features targets = BatchPlan
    { plannedBatchSizes :: Set.Set Int
    , batchLoads :: [m (SomeBatch in_sh out_sh features targets)]
    }

-- | Concatenate loading schedules and combine their expected sizes.
instance Semigroup (BatchPlan m in_sh out_sh features targets) where
    first <> second =
        BatchPlan
            { plannedBatchSizes = Set.union (plannedBatchSizes first) (plannedBatchSizes second)
            , batchLoads = batchLoads first ++ batchLoads second
            }

instance Monoid (BatchPlan m in_sh out_sh features targets) where
    mempty =
        BatchPlan
            { plannedBatchSizes = Set.empty
            , batchLoads = []
            }

{- | Describe a custom source with a finite collection of expected batch sizes.
Sizes must be nonnegative; duplicate sizes are ignored.
-}
batchPlan
    :: [Int]
    -> [m (SomeBatch in_sh out_sh features targets)]
    -> BatchPlan m in_sh out_sh features targets
batchPlan sizes loads
    | any (< 0) sizes = error "Synapse.NN.batchPlan: batch sizes must be nonnegative"
    | otherwise =
        BatchPlan
            { plannedBatchSizes = Set.fromList sizes
            , batchLoads = loads
            }

-- | Plan one pass over a dataset using the same size and remainder policy as 'loadBatches'.
planBatches
    :: ( Applicative m
       , Shape in_sh
       , Shape out_sh
       , forall batch. KnownNat batch => Shape (batch ': in_sh)
       , forall batch. KnownNat batch => Shape (batch ': out_sh)
       , DType features
       , DType targets
       )
    => Int
    -> LastBatch features targets
    -> Dataset m (TensorData in_sh features, TensorData out_sh targets)
    -> BatchPlan m in_sh out_sh features targets
planBatches limit policy source
    | limit <= 0 = error "Synapse.NN.planBatches: requested size must be positive"
    | datasetSize source < 0 = error "Synapse.NN.planBatches: dataset size must be nonnegative"
    | otherwise =
        let (full, remainder) = datasetSize source `quotRem` limit
            sizes = case policy of
                KeepLast -> [limit | full > 0] ++ [remainder | remainder > 0]
                DropLast -> [limit | full > 0]
                PadLast _ _ -> [limit | datasetSize source > 0]
         in batchPlan sizes $ loadBatches limit policy source

-- | Distinct expected batch sizes in ascending order, including padding.
batchSizes
    :: BatchPlan m in_sh out_sh features targets
    -> [Int]
batchSizes = Set.toAscList . plannedBatchSizes

{- | An immutable executor with a prepared function for each planned batch size.
Retain it across batches and epochs to reuse the prepared functions.
-}
newtype BatchStep args batch result = BatchStep
    { runBatch :: args -> batch -> Either UnexpectedSize result
    }

{- | The actual leading dimension was not included in the preparation plan.
The caller retains the original batch and can choose another execution path.
The stored value is the actual batch size.
-}
newtype UnexpectedSize = UnexpectedSize Int deriving (Eq, Show)

{- | Prepare one computation for all expected sizes in a loading plan to be executed on a specified backend.

The returned 'BatchStep' retains one function prepared by 'run' for each planned batch size.
Bind it outside the batch loop and reuse it across epochs to share preparation and compiled operations
between calls of the same size, including a smaller final batch.
Preparation is lazy; compilation may occur when a specialization is first used.

An unplanned size returns @Left (UnexpectedSize size)@ without preparing another specialization;
you may do that yourself if needed.

@
let plan = planBatches 32 KeepLast dataset
    step = prepareBatchStep defaultCPU plan trainingStep
forM_ (batchLoads plan) $ \\load -> do
  batch <- load
  case runBatch step parameters batch of
    Left unexpected -> ...
    Right result -> ...
@
-}
prepareBatchStep
    :: forall args result m in_sh out_sh features targets
     . ( RunnableTensor args
       , RunnableTensor result
       , forall batch. KnownNat batch => Shape (batch ': in_sh)
       , forall batch. KnownNat batch => Shape (batch ': out_sh)
       , DType features
       , DType targets
       )
    => Backend
    -> BatchPlan m in_sh out_sh features targets
    -> ( forall batch
          . KnownNat batch
         => args
         -> Tensor (batch ': in_sh) features
         -> Tensor (batch ': out_sh) targets
         -> Tensor '[batch] Bool
         -> result
       )
    -> BatchStep
        (WithTensor TensorData args)
        (SomeBatch in_sh out_sh features targets)
        (WithTensor TensorData result)
prepareBatchStep backend plan operation =
    let specialize size = case someNatVal $ fromIntegral size of
            SomeNat (proxy :: Proxy batch) ->
                let execute = run backend $ operation @batch
                 in Prepared proxy $ \args batch ->
                        execute args (batchInputs batch) (batchTargets batch) (batchMask batch)
        entries = Map.fromDistinctAscList [(size, specialize size) | size <- batchSizes plan]
     in BatchStep $ \args packed -> withBatch packed $ \(batch :: Batch n in_sh out_sh features targets) ->
            let size = fromIntegral $ natVal $ Proxy @n
             in case Map.lookup size entries of
                    Just (Prepared proxy execute) -> case sameNat (Proxy @n) proxy of
                        Just Refl -> Right $ execute args batch
                        Nothing -> Left $ UnexpectedSize size
                    Nothing -> Left $ UnexpectedSize size
