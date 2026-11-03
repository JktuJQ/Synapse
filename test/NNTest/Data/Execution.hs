{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Data.Execution (tests) where

import Control.Exception (ErrorCall, evaluate, try)
import Control.Monad (foldM, forM_)
import Data.Functor.Identity (Identity, runIdentity)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.List (nub, sort)
import GHC.TypeNats (KnownNat)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

source :: Int -> NN.Dataset Identity (T.TensorData '[1] Double, T.TensorData '[1] Double)
source count = NN.Indexed count $ \index ->
  let x = fromIntegral $ index `mod` 3 + 1
   in pure (T.fromListData [x], T.fromListData [2 * x])

objective ::
  forall batch_size.
  (KnownNat batch_size) =>
  T.Tensor '[] Double ->
  T.Tensor '[batch_size, 1] Double ->
  T.Tensor '[batch_size, 1] Double ->
  T.Tensor '[batch_size] Bool ->
  (T.Tensor '[] Double, T.Tensor '[] Double)
objective weight inputs targets mask =
  let weights = T.select (T.unsqueeze @1 mask) 1 0 :: T.Tensor '[batch_size, 1] Double
      loss :: (AG.Context s) => AG.Tensor s '[] Double -> AG.Tensor s '[] Double
      loss p =
        NN.mse
          (\errors -> T.sum (errors * AG.constant weights) / AG.constant (T.sum weights))
          (T.broadcast p * AG.constant inputs)
          (AG.constant targets)
   in AG.backprop loss weight

tests :: TestTree
tests =
  testGroup
    "batch preparation"
    [ testProperty "planned sizes match loaded batches for every remainder policy" $ H.property $ do
        count <- H.forAll $ Gen.int $ Range.linear 0 100
        limit <- H.forAll $ Gen.int $ Range.linear 1 20
        forM_ [NN.KeepLast, NN.DropLast, NN.PadLast 99999 (-12345)] $ \policy -> do
          let plan = NN.planBatches limit policy $ source count
              actual = map (\load -> NN.withBatch (runIdentity load) NN.batchSize) $ NN.batchLoads plan
          NN.batchSizes plan H.=== map fromIntegral (sort $ nub actual),
      testProperty "custom plans normalize sizes without inspecting their actions" $ H.withTests 1 $ H.property $ do
        let plan = NN.batchPlan [16, 3, 16] (error "loading schedule evaluated") :: NN.BatchPlan IO '[1] '[1] Double Double
            step = NN.prepareBatchStep T.CPU plan objective
            batch =
              NN.SomeBatch $
                NN.Batch
                  (T.fromListData @'[3, 1] [1, 2, 3 :: Double])
                  (T.fromListData @'[3, 1] [2, 4, 6 :: Double])
                  3
        NN.batchSizes plan H.=== [3, 16]
        case NN.runBatch step (T.fromListData @'[] [1]) batch of
          Left _ -> H.failure
          Right (loss, gradient) -> do
            approxList (T.toList loss) [14 / 3]
            approxList (T.toList gradient) [negate $ 28 / 3],
      testProperty "plans combine loading order and size specializations" $ H.withTests 1 $ H.property $ do
        let first = NN.planBatches 16 NN.KeepLast $ source 23
            second = NN.planBatches 5 NN.KeepLast $ source 8
            combined = first <> second
            step = NN.prepareBatchStep T.CPU combined objective
            weight = T.fromListData @'[] [1 :: Double]
            sizes :: NN.BatchPlan Identity '[1] '[1] Double Double -> [Int]
            sizes plan = map (\load -> NN.withBatch (runIdentity load) NN.batchSize) $ NN.batchLoads plan
        NN.batchSizes combined H.=== [3, 5, 7, 16]
        sizes combined H.=== sizes first ++ sizes second
        forM_ (NN.batchLoads combined) $ \load -> case NN.runBatch step weight $ runIdentity load of
          Left _ -> H.failure
          Right (loss, _) -> H.assert $ all (> 0) $ T.toList loss,
      testProperty "prepared gradients exclude padding and accept changing arguments across epochs" $ H.withTests 1 $ H.property $ do
        let kept = NN.planBatches 16 NN.KeepLast $ source 23
            padded = NN.planBatches 16 (NN.PadLast 99999 (-12345)) $ source 23
            step = NN.prepareBatchStep T.CPU (kept <> padded) objective
            initial = T.fromListData @'[] [1 :: Double]
            moved = T.fromListData @'[] [3 :: Double]
            expected first count =
              let xs = [fromIntegral $ index `mod` 3 + 1 | index <- [first .. first + count - 1 :: Int]]
               in sum (map (\x -> x * x) xs) / fromIntegral count
            squares = [expected 0 16, expected 16 7]
        forM_ [kept, padded, kept] $ \plan ->
          forM_ [(initial, -2), (moved, 2)] $ \(weight, factor) ->
            forM_ (zip (NN.batchLoads plan) squares) $ \(load, square) ->
              case NN.runBatch step weight $ runIdentity load of
                Left _ -> H.failure
                Right (loss, gradient) -> do
                  approxList (T.toList loss) [square]
                  approxList (T.toList gradient) [factor * square],
      testProperty "unknown sizes are reported and the caller can fall back to runN" $ H.withTests 1 $ H.property $ do
        let plan = NN.planBatches 16 NN.DropLast $ source 32
            step = NN.prepareBatchStep T.CPU plan objective
            other = map runIdentity $ NN.loadBatches 3 NN.DropLast $ source 3
            weight = T.fromListData @'[] [1 :: Double]
        forM_ other $ \packed -> case NN.runBatch step weight packed of
          Right _ -> H.failure
          Left unexpected -> do
            unexpected H.=== NN.UnexpectedSize 3
            let (loss, gradient) = NN.withBatch packed $ \batch ->
                  T.run4 T.CPU objective weight (NN.batchInputs batch) (NN.batchTargets batch) (NN.batchMask batch)
            approxList (T.toList loss) [14 / 3]
            approxList (T.toList gradient) [negate $ 28 / 3],
      testProperty "an empty executor rejects batches without evaluating its computation" $ H.withTests 1 $ H.property $ do
        let plan = mempty :: NN.BatchPlan Identity '[1] '[1] Double Double
            step = NN.prepareBatchStep T.CPU plan $ \(_ :: T.Tensor '[] Double) _ _ _ -> error "unexpected preparation" :: T.Tensor '[] Double
        NN.batchSizes plan H.=== []
        forM_ (NN.loadBatches 1 NN.KeepLast $ source 1) $ \load ->
          case NN.runBatch step (T.fromListData @'[] [1]) $ runIdentity load of
            Left err -> err H.=== NN.UnexpectedSize 1
            Right _ -> H.failure,
      testProperty "custom IO loading works with foldM and performs no reads during preparation" $ H.withTests 1 $ H.property $ do
        count <- H.evalIO $ newIORef (0 :: Int)
        let batch =
              NN.SomeBatch $
                NN.Batch
                  (T.fromListData @'[2, 1] [1, 2 :: Double])
                  (T.fromListData @'[2, 1] [2, 4 :: Double])
                  2
            load = modifyIORef' count (+ 1) >> pure batch
            plan = NN.batchPlan [2] [load, load]
            step = NN.prepareBatchStep T.CPU plan objective
        _ <- H.evalIO $ evaluate step
        before <- H.evalIO $ readIORef count
        before H.=== 0
        result <-
          H.evalIO $
            foldM
              ( \total action -> do
                  packed <- action
                  (loss, _) <- either (fail . show) pure $ NN.runBatch step (T.fromListData @'[] [1]) packed
                  pure $ total + sum (T.toList loss)
              )
              0
              (NN.batchLoads plan)
        approxList [result] [5]
        after <- H.evalIO $ readIORef count
        after H.=== 2,
      testProperty "planning rejects invalid sizes even for an empty source" $ H.withTests 1 $ H.property $ do
        forM_ [(0, 0), (-1, 0), (1, -1)] $ \(limit, count) -> do
          result <- H.evalIO $ try @ErrorCall $ evaluate $ NN.planBatches limit NN.KeepLast $ source count
          case result of
            Left _ -> pure ()
            Right _ -> H.failure
    ]
