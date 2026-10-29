{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module NNTest.Data (tests) where

import Control.Exception (ErrorCall, displayException, evaluate, try)
import Control.Monad (foldM, forM_)
import Data.Char (ord)
import Data.Functor.Identity (Identity, runIdentity)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.List (foldl', isInfixOf, sort)
import GHC.TypeNats (KnownNat)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import System.IO (IOMode (ReadMode), SeekMode (AbsoluteSeek), hGetChar, hSeek, withFile)
import System.Random (mkStdGen)
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

summary :: NN.SomeBatch '[] '[] Int Int -> ([Int], [Int], Int, Int, [Bool])
summary some = NN.withBatch some $ \batch ->
  (T.toList (NN.batchInputs batch), T.toList (NN.batchTargets batch), NN.batchSize batch, NN.validCount batch, T.toList (NN.batchMask batch))

sampleBatches ::
  (T.Shape in_sh, T.Shape out_sh, forall batch. (KnownNat batch) => T.Shape (batch ': in_sh), forall batch. (KnownNat batch) => T.Shape (batch ': out_sh), T.DType a, T.DType b) =>
  Int ->
  NN.LastBatch a b ->
  [(T.TensorData in_sh a, T.TensorData out_sh b)] ->
  [NN.SomeBatch in_sh out_sh a b]
sampleBatches size policy samples =
  let count = length samples
      source = NN.Indexed count (pure . (samples !!))
   in map runIdentity (NN.loadBatches size policy source)

step :: forall batch. (KnownNat batch) => NN.Dense 1 1 T.Tensor Double -> T.Tensor '[batch, 1] Double -> T.Tensor '[batch, 1] Double -> T.Tensor '[batch] Bool -> NN.Dense 1 1 T.Tensor Double
step parameters inputs targets mask =
  let weights = T.select (T.unsqueeze @1 mask) 1 0 :: T.Tensor '[batch, 1] Double
      objective :: forall s. (AG.Context s) => NN.Dense 1 1 (AG.Tensor s) Double -> AG.Tensor s '[] Double
      objective p = NN.mse (\errors -> T.sum (errors * AG.constant weights) / AG.constant (T.sum weights)) (NN.dense p (AG.constant inputs)) (AG.constant targets)
      gradients = AG.gradient objective parameters
   in fst (NN.optimizerStep (NN.SGD 0.01) parameters gradients ())

train :: NN.LastBatch Double Double -> [Double]
train policy =
  let samples = [(T.fromListData @'[1] [x], T.fromListData @'[1] [2 * x + 1]) | x <- [1 .. 5]]
      batches = sampleBatches 2 policy samples
      initial = T.run T.CPU (NN.Dense (T.zeros @'[1, 1]) (T.zeros @'[1]) :: NN.Dense 1 1 T.Tensor Double)
      update :: NN.Dense 1 1 T.TensorData Double -> NN.SomeBatch '[1] '[1] Double Double -> NN.Dense 1 1 T.TensorData Double
      update parameters some = NN.withBatch some $ \batch -> T.run4 T.CPU step parameters (NN.batchInputs batch) (NN.batchTargets batch) (NN.batchMask batch)
      final = foldl' update initial (concat (replicate 3 batches))
   in T.toList (NN.denseWeights final) ++ T.toList (NN.denseBias final)

tests :: TestTree
tests =
  testGroup
    "data"
    [ testProperty "policies preserve paired examples, sizes, and validity masks" $ H.property $ do
        count <- H.forAll $ Gen.int (Range.linear 0 50)
        limit <- H.forAll $ Gen.int (Range.linear 1 12)
        let samples = [(T.fromListData @'[] [i], T.fromListData @'[] [100 + i]) | i <- [1 .. count]]
        forM_ [NN.KeepLast, NN.DropLast, NN.PadLast (-1) (-2)] $ \policy -> do
          let batches = map summary (sampleBatches limit policy samples)
              expected = case policy of
                NN.DropLast -> count `div` limit * limit
                _ -> count
          concat [take valid xs | (xs, _, _, valid, _) <- batches] H.=== [1 .. expected]
          concat [take valid ys | (_, ys, _, valid, _) <- batches] H.=== [101 .. 100 + expected]
          forM_ batches $ \(xs, ys, size, valid, mask) -> do
            H.assert (valid > 0 && valid <= size && size <= limit)
            length xs H.=== size
            length ys H.=== size
            mask H.=== replicate valid True ++ replicate (size - valid) False
            case policy of
              NN.PadLast x y -> do
                size H.=== limit
                drop valid xs H.=== replicate (size - valid) x
                drop valid ys H.=== replicate (size - valid) y
              _ -> size H.=== valid,
      testProperty "inputs and targets retain different sample ranks and element types" $ H.withTests 1 $ H.property $ do
        let sample = (T.fromListData @'[2] [1, 2 :: Int], T.fromListData @'[2, 2] [True, False, False, True])
            batches = sampleBatches 2 (NN.PadLast (-1) False) [sample]
        length batches H.=== 1
        forM_ batches $ \some -> NN.withBatch some $ \batch -> do
          T.toList (NN.batchInputs batch) H.=== [1, 2, -1, -1]
          T.toList (NN.batchTargets batch) H.=== [True, False, False, True, False, False, False, False]
          T.toList (NN.batchMask batch) H.=== [True, False],
      testProperty "empty datasets and dropped short datasets produce no batches" $ H.withTests 1 $ H.property $ do
        let empty = [] :: [(T.TensorData '[] Int, T.TensorData '[] Int)]
            sample = (T.fromListData @'[] [1 :: Int], T.fromListData @'[] [2 :: Int])
        forM_ [NN.KeepLast, NN.DropLast, NN.PadLast 0 0] $ \policy ->
          length (sampleBatches 3 policy empty) H.=== 0
        length (sampleBatches 3 NN.DropLast [sample]) H.=== 0,
      testProperty "zero-sized sample dimensions retain the number of examples" $ H.withTests 1 $ H.property $ do
        let sample = (T.fromListData @'[0] ([] :: [Int]), T.fromListData @'[] [True])
            batches = sampleBatches 3 (NN.PadLast 0 False) [sample]
        forM_ batches $ \some -> NN.withBatch some $ \batch -> do
          NN.batchSize batch H.=== 3
          NN.validCount batch H.=== 1
          T.toList (NN.batchInputs batch) H.=== []
          T.toList (NN.batchTargets batch) H.=== [True, False, False],
      testProperty "a large pure source loads only the consumed prefix" $ H.withTests 1 $ H.property $ do
        let source :: NN.Dataset Identity (T.TensorData '[] Int, T.TensorData '[] Int)
            source = NN.Indexed 1000000000 $ \i ->
              if i < 6 then pure (T.fromListData @'[] [i], T.fromListData @'[] [100 + i]) else error "unexpected sample read"
            first = take 2 (map runIdentity (NN.loadBatches 3 NN.KeepLast source))
        map summary first H.=== [([0, 1, 2], [100, 101, 102], 3, 3, [True, True, True]), ([3, 4, 5], [103, 104, 105], 3, 3, [True, True, True])],
      testProperty "nonpositive batch sizes are rejected even for an empty dataset" $ H.withTests 1 $ H.property $ do
        forM_ [0, -1] $ \size -> do
          let batches = sampleBatches size NN.KeepLast [] :: [NN.SomeBatch '[] '[] Int Int]
          result <- H.evalIO $ try @ErrorCall (evaluate batches)
          case result of
            Left _ -> pure ()
            Right _ -> H.failure,
      testProperty "batch records can be constructed, updated, and unpacked directly" $ H.withTests 1 $ H.property $ do
        let original = NN.Batch {NN.batchInputs = T.fromListData @'[2] [3, 4 :: Int], NN.batchTargets = T.fromListData @'[2] [30, 40 :: Int], NN.validCount = 2}
            padded = original {NN.validCount = 1}
            packed = NN.SomeBatch padded
        summary packed H.=== ([3, 4], [30, 40], 2, 1, [True, False])
        case packed of
          NN.SomeBatch batch -> NN.validCount batch H.=== 1,
      testProperty "shuffle permutes dataset indices without reading examples" $ H.property $ do
        count <- H.forAll $ Gen.int (Range.linear 0 100)
        seed <- H.forAll $ Gen.int (Range.linear (-10000) 10000)
        requests <- H.evalIO $ newIORef []
        let dataset = NN.Indexed count (\i -> modifyIORef' requests (i :) >> pure (T.fromListData @'[] [i], T.fromListData @'[] [i :: Int]))
            (shuffled, _) = NN.shuffle dataset (mkStdGen seed)
        NN.datasetSize shuffled H.=== count
        before <- H.evalIO $ readIORef requests
        before H.=== []
        values <- H.evalIO $ mapM (NN.getSample shuffled) [0 .. count - 1]
        sort (concatMap (T.toList . fst) values) H.=== [0 .. count - 1]
        after <- H.evalIO $ readIORef requests
        reverse after H.=== concatMap (T.toList . fst) values,
      testProperty "shuffle is reproducible and advances the generator for subsequent epochs" $ H.withTests 1 $ H.property $ do
        let dataset = NN.Indexed 32 (\i -> pure (T.fromListData @'[] [i], T.fromListData @'[] [i :: Int]))
            generator = mkStdGen 19
            (first, next) = NN.shuffle dataset generator
            (repeated, repeatedNext) = NN.shuffle dataset generator
            (second, _) = NN.shuffle dataset next
            order source = concatMap (T.toList . fst) <$> mapM (NN.getSample source) [0 .. NN.datasetSize source - 1]
        firstOrder <- H.evalIO $ order first
        repeatedOrder <- H.evalIO $ order repeated
        secondOrder <- H.evalIO $ order second
        firstOrder H.=== repeatedOrder
        next H.=== repeatedNext
        H.assert (next /= generator)
        H.assert (secondOrder /= firstOrder),
      testProperty "shuffle supports empty and singleton datasets" $ H.withTests 1 $ H.property $ do
        forM_ [0, 1] $ \count -> do
          let (dataset, _) = NN.shuffle (NN.Indexed count (\i -> pure (T.fromListData @'[] [i], T.fromListData @'[] [i :: Int]))) (mkStdGen 1)
          values <- H.evalIO $ mapM (NN.getSample dataset) [0 .. count - 1]
          concatMap (T.toList . fst) values H.=== [0 .. count - 1],
      testProperty "shuffling and loading batches preserves paired examples" $ H.withTests 1 $ H.property $ do
        let dataset = NN.Indexed 17 (\i -> pure (T.fromListData @'[] [i], T.fromListData @'[] [100 + i :: Int]))
            (shuffled, _) = NN.shuffle dataset (mkStdGen 23)
        loaded <- H.evalIO $ sequence (NN.loadBatches 4 NN.KeepLast shuffled)
        let batches = map summary loaded
            inputs = concat [xs | (xs, _, _, _, _) <- batches]
            targets = concat [ys | (_, ys, _, _, _) <- batches]
        sort inputs H.=== [0 .. 16]
        targets H.=== map (+ 100) inputs
        map (\(_, _, size, _, _) -> size) batches H.=== [4, 4, 4, 4, 1],
      testProperty "loading is incremental and dropped examples are never read" $ H.withTests 1 $ H.property $ do
        requests <- H.evalIO $ newIORef []
        let dataset = NN.Indexed 5 (\i -> modifyIORef' requests (i :) >> pure (T.fromListData @'[] [i], T.fromListData @'[] [100 + i :: Int]))
            loads = NN.loadBatches 2 NN.DropLast dataset
        length loads H.=== 2
        before <- H.evalIO $ readIORef requests
        before H.=== []
        case loads of
          [first, second] -> do
            batch <- H.evalIO first
            summary batch H.=== ([0, 1], [100, 101], 2, 2, [True, True])
            afterFirst <- H.evalIO $ readIORef requests
            reverse afterFirst H.=== [0, 1]
            _ <- H.evalIO second
            afterSecond <- H.evalIO $ readIORef requests
            reverse afterSecond H.=== [0, 1, 2, 3]
          _ -> H.failure,
      testProperty "effectful and pure sources agree for every final-batch policy" $ H.withTests 1 $ H.property $ do
        let sample i = (T.fromListData @'[] [i], T.fromListData @'[] [100 + i :: Int])
            dataset = NN.Indexed 5 (pure . sample)
        forM_ [NN.KeepLast, NN.DropLast, NN.PadLast (-1) (-2)] $ \policy -> do
          loaded <- H.evalIO $ sequence (NN.loadBatches 2 policy dataset)
          map summary loaded H.=== map summary (sampleBatches 2 policy (map sample [0 .. 4])),
      testProperty "file-backed sources read samples only when a batch is executed" $ H.withTests 1 $ H.property $ do
        requests <- H.evalIO $ newIORef []
        let dataset = NN.Indexed 3 $ \index -> do
              modifyIORef' requests (index :)
              value <- withFile "synapse.cabal" ReadMode $ \handle -> do
                hSeek handle AbsoluteSeek (fromIntegral index)
                ord <$> hGetChar handle
              pure (T.fromListData @'[] [value], T.fromListData @'[] [index :: Int])
            loads = NN.loadBatches 2 NN.KeepLast dataset
        before <- H.evalIO $ readIORef requests
        before H.=== []
        case loads of
          first : _ -> do
            batch <- H.evalIO first
            summary batch H.=== (map ord "ca", [0, 1], 2, 2, [True, True])
            after <- H.evalIO $ readIORef requests
            reverse after H.=== [0, 1]
          _ -> H.failure,
      testProperty "loading failures propagate without reading later samples" $ H.withTests 1 $ H.property $ do
        requests <- H.evalIO $ newIORef []
        let dataset = NN.Indexed 4 $ \i -> do
              modifyIORef' requests (i :)
              if i == 1 then ioError (userError "decode failed") else pure (T.fromListData @'[] [i], T.fromListData @'[] [i :: Int])
        case NN.loadBatches 3 NN.KeepLast dataset of
          first : _ -> do
            result <- H.evalIO $ try @IOError first
            case result of
              Left _ -> pure ()
              Right _ -> H.failure
            after <- H.evalIO $ readIORef requests
            reverse after H.=== [0, 1]
          _ -> H.failure,
      testProperty "grouped sources issue one request per batch and skip a dropped remainder" $ H.withTests 1 $ H.property $ do
        requests <- H.evalIO $ newIORef []
        let sample i = (T.fromListData @'[] [i], T.fromListData @'[] [100 + i :: Int])
            source = NN.Grouped {NN.datasetSize = 5, NN.readSamples = \indices -> modifyIORef' requests (indices :) >> pure (map sample indices)}
            loads = NN.loadBatches 2 NN.DropLast source
        length loads H.=== 2
        before <- H.evalIO $ readIORef requests
        before H.=== []
        case loads of
          [first, second] -> do
            loaded <- H.evalIO first
            summary loaded H.=== ([0, 1], [100, 101], 2, 2, [True, True])
            afterFirst <- H.evalIO $ readIORef requests
            reverse afterFirst H.=== [[0, 1]]
            _ <- H.evalIO second
            afterSecond <- H.evalIO $ readIORef requests
            reverse afterSecond H.=== [[0, 1], [2, 3]]
          _ -> H.failure,
      testProperty "grouped and indexed sources agree after shuffle for all remainder policies" $ H.property $ do
        count <- H.forAll $ Gen.int (Range.linear 0 20)
        limit <- H.forAll $ Gen.int (Range.linear 1 8)
        seed <- H.forAll $ Gen.int (Range.linear (-1000) 1000)
        let sample i = (T.fromListData @'[2] [i, -i], T.fromListData @'[2, 1] [100 + i, 200 + i :: Int])
            indexed = NN.Indexed {NN.datasetSize = count, NN.readSample = pure . sample}
            grouped = NN.Grouped {NN.datasetSize = count, NN.readSamples = pure . map sample}
            contents :: NN.SomeBatch '[2] '[2, 1] Int Int -> ([Int], [Int], Int, [Bool])
            contents some = NN.withBatch some $ \batch -> (T.toList (NN.batchInputs batch), T.toList (NN.batchTargets batch), NN.validCount batch, T.toList (NN.batchMask batch))
        forM_ [NN.KeepLast, NN.DropLast, NN.PadLast (-1) (-2)] $ \policy -> do
          let (left, _) = NN.shuffle indexed (mkStdGen seed)
              (right, _) = NN.shuffle grouped (mkStdGen seed)
          map (contents . runIdentity) (NN.loadBatches limit policy left) H.=== map (contents . runIdentity) (NN.loadBatches limit policy right),
      testProperty "shuffle remaps grouped requests together without reading examples" $ H.withTests 1 $ H.property $ do
        requests <- H.evalIO $ newIORef []
        let sample i = (T.fromListData @'[] [i], T.fromListData @'[] [100 + i :: Int])
            source = NN.Grouped 5 (\indices -> modifyIORef' requests (indices :) >> pure (map sample indices))
            (shuffled, _) = NN.shuffle source (mkStdGen 19)
            loads = NN.loadBatches 2 NN.KeepLast shuffled
        length loads H.=== 3
        before <- H.evalIO $ readIORef requests
        before H.=== []
        loaded <- H.evalIO $ sequence loads
        let returned = concat [xs | (xs, _, _, _, _) <- map summary loaded]
        after <- H.evalIO $ readIORef requests
        map length (reverse after) H.=== [2, 2, 1]
        concat (reverse after) H.=== returned
        sort returned H.=== [0 .. 4],
      testProperty "datasets remain generic and support single-example access in both modes" $ H.withTests 1 $ H.property $ do
        let indexed = NN.Indexed 4 (pure . (* 10)) :: NN.Dataset Identity Int
            grouped = NN.Grouped 4 (pure . map (* 10)) :: NN.Dataset Identity Int
        forM_ [indexed, grouped] $ \source -> do
          NN.datasetSize source H.=== 4
          runIdentity (NN.getSample source 2) H.=== 20
        runIdentity (NN.readSamples grouped [3, 1, 3]) H.=== [30, 10, 30],
      testProperty "grouped sources support zero-sized samples and empty datasets" $ H.withTests 1 $ H.property $ do
        let sample _ = (T.fromListData @'[0] ([] :: [Int]), T.fromListData @'[] [True])
            source = NN.Grouped 2 (pure . map sample)
            empty = NN.Grouped 0 (\_ -> error "empty source must not be read") :: NN.Dataset Identity (T.TensorData '[0] Int, T.TensorData '[] Bool)
        length (NN.loadBatches 3 NN.KeepLast empty) H.=== 0
        let loaded = map runIdentity (NN.loadBatches 3 (NN.PadLast 0 False) source)
        forM_ loaded $ \some -> NN.withBatch some $ \batch -> do
          T.toList (NN.batchInputs batch) H.=== []
          T.toList (NN.batchTargets batch) H.=== [True, True, False]
          NN.validCount batch H.=== 2,
      testProperty "grouped length checks stop at the first excess example" $ H.withTests 1 $ H.property $ do
        let source = NN.Grouped 1 (\_ -> pure (1 : 2 : error "unexpected traversal")) :: NN.Dataset Identity Int
        result <- H.evalIO $ try @ErrorCall $ evaluate (runIdentity (NN.getSample source 0))
        case result of
          Left err -> H.assert ("grouped reader" `isInfixOf` displayException err)
          Right _ -> H.failure,
      testProperty "grouped readers returning too few or too many examples are rejected" $ H.withTests 1 $ H.property $ do
        let sample = (T.fromListData @'[] [1 :: Int], T.fromListData @'[] [2 :: Int])
        forM_ [-1, 1] $ \difference -> do
          let source = NN.Grouped 3 (\indices -> pure (replicate (length indices + difference) sample))
          case NN.loadBatches 2 NN.KeepLast source of
            first : _ -> do
              result <- H.evalIO $ try @ErrorCall $ do
                batch <- first
                evaluate (sum (NN.withBatch batch (T.toList . NN.batchInputs)))
              case result of
                Left _ -> pure ()
                Right _ -> H.failure
            _ -> H.failure,
      testProperty "on-demand batches feed staged training without loading an entire epoch" $ H.withTests 1 $ H.property $ do
        let dataset = NN.Indexed 5 (\i -> let x = fromIntegral (i + 1) in pure (T.fromListData @'[1] [x], T.fromListData @'[1] [2 * x + 1 :: Double]))
            initial = T.run T.CPU (NN.Dense (T.zeros @'[1, 1]) (T.zeros @'[1]) :: NN.Dense 1 1 T.Tensor Double)
            update :: NN.Dense 1 1 T.TensorData Double -> IO (NN.SomeBatch '[1] '[1] Double Double) -> IO (NN.Dense 1 1 T.TensorData Double)
            update parameters load = do
              some <- load
              NN.withBatch some $ \batch -> do
                let updated = T.run4 T.CPU step parameters (NN.batchInputs batch) (NN.batchTargets batch) (NN.batchMask batch)
                _ <- evaluate (sum (T.toList (NN.denseWeights updated)) + sum (T.toList (NN.denseBias updated)))
                pure updated
        final <- H.evalIO $ foldM update initial (concat (replicate 3 (NN.loadBatches 2 NN.KeepLast dataset)))
        approxList (T.toList (NN.denseWeights final) ++ T.toList (NN.denseBias final)) (train NN.KeepLast),
      testProperty "masked padding and a smaller final batch produce the same SGD updates" $ H.withTests 1 $ H.property $ do
        approxList (train (NN.PadLast 0 0)) (train NN.KeepLast)
    ]
