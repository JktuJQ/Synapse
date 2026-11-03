{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Attention (tests) where

import Control.Arrow (arr)
import Control.Monad (foldM)
import Data.List (transpose)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

rows :: Int -> [a] -> [[a]]
rows _ [] = []
rows width xs = let (first, rest) = splitAt width xs in first : rows width rest

-- Independent host reference: one head, arbitrary query/key/value widths.
reference :: Maybe [[Bool]] -> [[Double]] -> [[Double]] -> [[Double]] -> [[Double]]
reference mask query key value =
  [ let allowed = maybe (replicate (length key) True) (!! index) mask
        scores = [sum (zipWith (*) q k) / sqrt (fromIntegral $ length q) | k <- key]
        validScores = [score | (score, True) <- zip scores allowed]
        maximumScore = maximum $ 0 : validScores
        numerators = zipWith (\score valid -> if valid then exp (score - maximumScore) else 0) scores allowed
        denominator = sum numerators
        weights = if denominator == 0 then numerators else map (/ denominator) numerators
     in [sum $ zipWith (*) weights column | column <- transpose value]
    | (index, q) <- zip [0 ..] query
  ]

finiteDifferences :: ([Double] -> Double) -> [Double] -> [Double]
finiteDifferences objective inputs =
  [ (objective (change epsilon index) - objective (change (-epsilon) index)) / (2 * epsilon)
    | index <- [0 .. length inputs - 1]
  ]
  where
    epsilon = 1e-5
    change delta index = [if position == index then value + delta else value | (position, value) <- zip [0 ..] inputs]

parameterValues :: [Double]
parameterValues = [0.1 * sin (fromIntegral index) | index <- [1 .. 24 :: Int]]

parameters :: [Double] -> NN.MultiHeadAttention 2 2 T.Tensor Double
parameters xs = case rows 6 xs of
  [q, k, v, o] -> NN.MultiHeadAttention (projection q) (projection k) (projection v) (projection o)
  _ -> error "attention test parameter count"
  where
    projection x = NN.Dense (T.fromList $ take 4 x) (T.fromList $ drop 4 x)

flattenParameters :: NN.MultiHeadAttention 2 2 T.Tensor Double -> [Double]
flattenParameters p = concatMap (\dense -> values (NN.denseWeights dense) ++ values (NN.denseBias dense)) [NN.attentionQuery p, NN.attentionKey p, NN.attentionValue p, NN.attentionOutput p]

referenceMultiHead :: [Double] -> [[Double]] -> [[Double]] -> [[Double]] -> [[Double]]
referenceMultiHead weights query key value = case rows 6 weights of
  [q, k, v, o] ->
    let projectedQ = project q query
        projectedK = project k key
        projectedV = project v value
        headOutput headIndex = reference Nothing (map (\row -> [row !! headIndex]) projectedQ) (map (\row -> [row !! headIndex]) projectedK) (map (\row -> [row !! headIndex]) projectedV)
        combined = zipWith (++) (headOutput 0) (headOutput 1)
     in project o combined
  _ -> error "attention host parameter count"
  where
    project [w00, w01, w10, w11, b0, b1] = map point
      where
        point [x, y] = [x * w00 + y * w10 + b0, x * w01 + y * w11 + b1]
        point _ = error "attention host feature count"
    project _ = error "attention host projection size"

tests :: TestTree
tests =
  testGroup
    "attention"
    [ testProperty "scaled dot product agrees with an independent rectangular reference" $ H.withTests 1 $ H.property $ do
        let q = [0.2, -0.4, 0.7, 0.3]
            k = [0.5, 0.6, -0.2, 0.1, 0.3, -0.7]
            v = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            result = NN.attention Nothing (T.fromList @'[1, 1, 2, 2] q) (T.fromList @'[1, 1, 3, 2] k) (T.fromList @'[1, 1, 3, 3] v)
        approxList (values result) $ concat $ reference Nothing (rows 2 q) (rows 2 k) (rows 3 v),
      testProperty "query key and value pullbacks agree with finite differences and nonuniform seeds" $ H.withTests 1 $ H.property $ do
        let q = [0.2, -0.4, 0.7, 0.3]
            k = [0.5, 0.6, -0.2, 0.1, 0.3, -0.7]
            v = [1, 2, 3, 4, 5, 6]
            seed = [0.3, -0.4, 0.5, 0.8]
            mask = [[True, False, True], [False, True, True]]
            (_, backward) =
              AG.pullback
                (\(query, key, value) -> NN.attention (Just $ T.fromList $ concat mask) query key value)
                (T.fromList @'[1, 1, 2, 2] q, T.fromList @'[1, 1, 3, 2] k, T.fromList @'[1, 1, 3, 2] v)
            (dq, dk, dv) = backward $ T.fromList seed
            objective xs =
              let (query, rest) = splitAt 4 xs; (key, value) = splitAt 6 rest
               in sum $ zipWith (*) seed $ concat $ reference (Just mask) (rows 2 query) (rows 2 key) (rows 2 value)
        approxList (values dq ++ values dk ++ values dv) $ finiteDifferences objective $ q ++ k ++ v,
      testProperty "causal and padding masks compose and ignore future and padded values" $ H.withTests 1 $ H.property $ do
        let causal = T.broadcast (NN.causalMask @3 @3) :: T.Tensor '[1, 1, 3, 3] Bool
            padding = T.broadcast $ NN.paddingMask $ T.fromList @'[1, 3] [True, True, False]
            allowed = causal T.&& padding
            q = T.zeros @'[1, 1, 3, 2] @Double
            v = T.fromList @'[1, 1, 3, 1] [2, 6, 100 :: Double]
            (output, (_, _, gradient)) = AG.backprop (\(query, key, value) -> NN.attention (Just allowed) query key value) (q, q, v)
        T.toList (T.run T.CPU $ NN.causalMask @2 @3) H.=== [True, False, False, True, True, False]
        approxList (values output) [2, 4, 4]
        approxList (values gradient) [2, 1, 0],
      testProperty "fully masked rows give finite zero outputs and gradients even for huge scores" $ H.withTests 1 $ H.property $ do
        let q = T.full @'[1, 1, 2, 1] (1000 :: Double)
            k = T.fromList @'[1, 1, 2, 1] [1000, -1000]
            v = T.fromList @'[1, 1, 2, 1] [7, 11]
            mask = T.full @'[1, 1, 2, 2] False
            (output, (dq, dk, dv)) = AG.backprop (\(query, key, value) -> NN.attention (Just mask) query key value) (q, k, v)
        mapM_ (\x -> approxList (values x) [0, 0]) [output, dq, dk, dv],
      testProperty "mixed empty rows preserve the valid row and exclude masked extreme scores" $ H.withTests 1 $ H.property $ do
        let q = T.full @'[1, 1, 2, 1] (1000 :: Double)
            k = T.fromList @'[1, 1, 2, 1] [1000, -1000]
            v = T.fromList @'[1, 1, 2, 1] [7, 11]
            mask = T.fromList @'[1, 1, 2, 2] [False, False, False, True]
            (output, (dq, dk, dv)) = AG.backprop (\(query, key, value) -> NN.attention (Just mask) query key value) (q, k, v)
        approxList (values output) [0, 11]
        approxList (values dq) [0, 0]
        approxList (values dk) [0, 0]
        approxList (values dv) [0, 1],
      testProperty "prepared attention accepts changing masks and Float inputs" $ H.withTests 1 $ H.property $ do
        let execute = T.run4 T.CPU $ \mask query key value -> NN.attention (Just mask) query key value
            q = T.fromListData @'[1, 1, 1, 1] [1 :: Float]
            k = T.fromListData @'[1, 1, 2, 1] [1000, -1000 :: Float]
            v = T.fromListData @'[1, 1, 2, 1] [3, 7 :: Float]
        T.toList (execute (T.fromListData @'[1, 1, 1, 2] [True, True]) q k v) H.=== [3]
        T.toList (execute (T.fromListData @'[1, 1, 1, 2] [False, True]) q k v) H.=== [7]
        T.toList (execute (T.fromListData @'[1, 1, 1, 2] [False, False]) q k v) H.=== [0],
      testProperty "empty queries and batches preserve output and gradient shapes" $ H.withTests 1 $ H.property $ do
        let q = T.zeros @'[1, 2, 0, 3] @Double
            k = T.ones @'[1, 2, 4, 3]
            v = T.ones @'[1, 2, 4, 2]
            (output, (_, dk, dv)) = AG.backprop (\(query, key, value) -> NN.attention Nothing query key value) (q, k, v)
        values output H.=== []
        values dk H.=== replicate 24 0
        values dv H.=== replicate 16 0
        values (NN.attention Nothing (T.zeros @'[0, 2, 3, 4] @Double) (T.zeros @'[0, 2, 5, 4]) (T.zeros @'[0, 2, 5, 2])) H.=== [],
      testProperty "multi-head projections and all parameter gradients match a host implementation" $ H.withTests 1 $ H.property $ do
        let q = [[0.2, -0.4], [0.7, 0.3]]
            k = [[0.5, 0.6], [-0.2, 0.1], [0.3, -0.7]]
            v = [[1, 2], [3, 4], [5, 6]]
            seed = [0.3, -0.4, 0.5, 0.8]
            (output, backward) = AG.pullback (\p -> NN.multiHeadAttention p Nothing (AG.constant $ T.fromList @'[1, 2, 2] $ concat q, AG.constant $ T.fromList @'[1, 3, 2] $ concat k, AG.constant $ T.fromList @'[1, 3, 2] $ concat v)) $ parameters parameterValues
            gradient = backward $ T.fromList seed
            objective p = sum $ zipWith (*) seed $ concat $ referenceMultiHead p q k v
        approxList (values output) $ concat $ referenceMultiHead parameterValues q k v
        approxList (flattenParameters gradient) $ finiteDifferences objective parameterValues,
      testProperty "multi-head input gradients propagate through all three projections" $ H.withTests 1 $ H.property $ do
        let xs = [0.2, -0.4, 0.7, 0.3, 0.5, 0.6, -0.2, 0.1, 0.3, -0.7, 1, 2, 3, 4, 5, 6]
            q = T.fromList @'[1, 2, 2] $ take 4 xs
            k = T.fromList @'[1, 3, 2] $ take 6 $ drop 4 xs
            v = T.fromList @'[1, 3, 2] $ drop 10 xs
            (_, (dq, dk, dv)) = AG.backprop (NN.multiHeadAttention (AG.constant $ parameters parameterValues) Nothing) (q, k, v)
            objective inputs =
              let (query, rest) = splitAt 4 inputs; (key, value) = splitAt 6 rest
               in sum $ concat $ referenceMultiHead parameterValues (rows 2 query) (rows 2 key) (rows 2 value)
        approxList (values dq ++ values dk ++ values dv) $ finiteDifferences objective xs,
      testProperty "multi-head self-attention composes in Forward and weights round-trip" $ H.withTests 1 $ H.property $ do
        let p = parameters parameterValues
            x = T.fromList @'[1, 2, 2] [0.2, -0.4, 0.7, 0.3]
            graph = proc input -> do
              inputs <- arr (\a -> (a, a, a)) -< input
              NN.multiHeadAttention p Nothing -< inputs
            (output, state) = NN.runState (NN.forward graph x) True
        state H.=== True
        approxList (values output) $ values $ NN.multiHeadAttention p Nothing (x, x, x)
        case NN.loadWeights (NN.extractWeights $ T.run T.CPU p) (T.run T.CPU $ parameters $ replicate 24 0) of
          Left failure -> H.annotateShow failure >> H.failure
          Right restored ->
            let actual = T.run1 T.CPU (\weights -> NN.multiHeadAttention weights Nothing (x, x, x)) restored
             in approxList (T.toList actual) $ values output,
      testProperty "heads with multiple features preserve batch and feature ordering" $ H.withTests 1 $ H.property $ do
        let projection = NN.Dense (T.eye @4 @4) 0
            p = NN.MultiHeadAttention projection projection projection projection :: NN.MultiHeadAttention 4 2 T.Tensor Double
            input = [0.2, -0.4, 0.7, 0.3, 0.5, 0.6, -0.2, 0.1, 1, 2, 3, 4, 5, 6, 7, 8]
            x = T.fromList @'[2, 2, 4] input
            expectedBatch batch =
              let first = map (take 2) batch
                  second = map (drop 2) batch
               in concat $ zipWith (++) (reference Nothing first first first) (reference Nothing second second second)
        approxList (values $ NN.multiHeadAttention p Nothing (x, x, x)) $ concatMap (expectedBatch . rows 4) $ rows 8 input,
      testProperty "prepared SGD updates attention parameters and reduces loss" $ H.withTests 1 $ H.property $ do
        let x = T.fromListData @'[1, 2, 2] [0.2, -0.4, 0.7, 0.3 :: Double]
            step = T.run2 T.CPU $ \p input ->
              let (loss, gradient) = AG.backprop (\weights -> T.sum $ let result = NN.multiHeadAttention weights Nothing (AG.constant input, AG.constant input, AG.constant input) in result * result) p
               in (loss, NN.evalState (NN.optimizerStep (NN.SGD 0.1) p gradient) ())
            initial = T.run T.CPU $ parameters parameterValues
            (firstLoss, _) = step initial x
        trained <- foldM (\p _ -> pure $ snd $ step p x) initial [1 .. 5 :: Int]
        let (lastLoss, _) = step trained x
        H.assert $ sum (T.toList lastLoss) < sum (T.toList firstLoss) / 10,
      testGroup
        "shape checks"
        [ testProperty name $ H.withTests 1 $ Compiler.rejects ("test/NNTest/Fixtures/Rejected/" ++ name ++ ".hs") "Couldn't match type"
          | name <- ["AttentionHeads", "AttentionDepth", "AttentionKeys"]
        ]
    ]
