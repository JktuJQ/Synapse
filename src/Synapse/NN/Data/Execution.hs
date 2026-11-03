{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Reuse prepared backend computations across runtime-selected batch sizes.
-- Each executor belongs to one computation and backend. Parameters, optimizer
-- state, and other changing tensor inputs are supplied on every invocation.
module Synapse.NN.Data.Execution
  ( PreparedStep,
    prepareBatchStep,
    runBatch,
    UnexpectedSize (..),
  )
where

import qualified Data.Map.Lazy as Map
import Data.Proxy (Proxy (..))
import Data.Type.Equality ((:~:) (Refl))
import GHC.TypeNats (KnownNat, SomeNat (..), natVal, sameNat, someNatVal)
import Numeric.Natural (Natural)
import Synapse.NN.Data (Batch (..), BatchPlan, SomeBatch, batchMask, batchSizes, withBatch)
import Synapse.Tensor.Backend (Backend, Computation, run4)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (DType, Tensor, TensorData, WithTensor)

-- | An immutable executor with a prepared function for each planned batch size.
-- Retain it across batches and epochs to reuse backend preparation. The executor
-- receives changing model parameters and optimizer state as runtime arguments.
-- Result shapes must be independent of the hidden batch size.
newtype PreparedStep args batch result
  = PreparedStep (args -> batch -> Either UnexpectedSize result)

-- | The actual leading dimension was not included in the preparation plan.
-- The batch remains available to the caller for an alternative execution path.
newtype UnexpectedSize = UnexpectedSize Natural deriving (Eq, Show)

data Prepared args in_sh out_sh features targets result
  = forall batch. (KnownNat batch) => Prepared
      (Proxy batch)
      (args -> Batch batch in_sh out_sh features targets -> result)

-- | Prepare one computation for all expected sizes in a loading plan.
-- The first computation argument packs changing tensor inputs such as parameters
-- and optimizer state; the last argument is the validity mask for padded batches.
-- The loading actions are not inspected or executed. Backend preparation is lazy:
-- a specialization may compile when its result is first demanded.
-- Values captured by the computation remain fixed across invocations; pass
-- changing values through its arguments instead.
--
-- @
-- let plan = planBatches 32 KeepLast dataset
--     step = prepareBatchStep CPU plan trainingStep
-- forM_ (batchLoads plan) $ \load -> do
--   batch <- load
--   case runBatch step parameters batch of
--     Left unexpected -> ...
--     Right result -> ...
-- @
prepareBatchStep ::
  forall args result m in_sh out_sh features targets.
  ( Computation args,
    Computation result,
    forall batch. (KnownNat batch) => Shape (batch ': in_sh),
    forall batch. (KnownNat batch) => Shape (batch ': out_sh),
    DType features,
    DType targets
  ) =>
  Backend ->
  BatchPlan m in_sh out_sh features targets ->
  ( forall batch.
    (KnownNat batch) =>
    args ->
    Tensor (batch ': in_sh) features ->
    Tensor (batch ': out_sh) targets ->
    Tensor '[batch] Bool ->
    result
  ) ->
  PreparedStep
    (WithTensor TensorData args)
    (SomeBatch in_sh out_sh features targets)
    (WithTensor TensorData result)
prepareBatchStep backend plan operation =
  let specialize size = case someNatVal size of
        SomeNat (proxy :: Proxy batch) ->
          let execute = run4 backend $ operation @batch
           in Prepared proxy $ \args batch ->
                execute args (batchInputs batch) (batchTargets batch) (batchMask batch)
      entries = Map.fromDistinctAscList [(size, specialize size) | size <- batchSizes plan]
   in PreparedStep $ \args packed -> withBatch packed $ \(batch :: Batch n in_sh out_sh features targets) ->
        let size = natVal $ Proxy @n
         in case Map.lookup size entries of
              Just (Prepared proxy execute) -> case sameNat (Proxy @n) proxy of
                Just Refl -> Right $ execute args batch
                Nothing -> Left $ UnexpectedSize size
              Nothing -> Left $ UnexpectedSize size

-- | Execute a batch using its saved specialization and fresh runtime arguments.
-- An unplanned size returns @Left (UnexpectedSize size)@; no new function is prepared and
-- the executor remains unchanged. Losses must explicitly use the supplied mask
-- when padded examples should be excluded.
runBatch :: PreparedStep args batch result -> args -> batch -> Either UnexpectedSize result
runBatch (PreparedStep execute) = execute
