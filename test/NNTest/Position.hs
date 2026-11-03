{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Position (tests) where

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

-- Independent host rotation of one head's sequence.
rotate :: Double -> Int -> [[Double]] -> [[Double]]
rotate base offset = zipWith rotateRow [offset ..]
  where
    rotateRow position input = pairs 0 input
      where
        width = fromIntegral $ length input
        pairs index (x : y : rest) =
          let angle = fromIntegral position / base ** (fromIntegral index / width)
           in (x * cos angle - y * sin angle) : (x * sin angle + y * cos angle) : pairs (index + 2 :: Int) rest
        pairs _ [] = []
        pairs _ _ = error "odd host rotation width"

tests :: TestTree
tests =
  testGroup
    "positions"
    [ testProperty "sinusoidal positions match formulas including odd widths and offsets" $ H.withTests 1 $ H.property $ do
        let expected = [if even feature then sin angle else cos angle | position <- [3 .. 5 :: Int], feature <- [0 .. 4 :: Int], let angle = fromIntegral position / (10000 ** (fromIntegral (2 * (feature `div` 2)) / 5))]
        approxList (values $ NN.sinusoidalPositions @3 @5 3) expected
        approxList (values $ NN.sinusoidalPositions @1 @3 0) [0, 1, 0]
        approxList (values $ NN.sinusoidalPositions @2 @1 (-1)) [sin (-1), 0]
        values (NN.sinusoidalPositions @0 @3 0 :: T.Tensor '[0, 3] Double) H.=== []
        values (NN.sinusoidalPositions @3 @0 0 :: T.Tensor '[3, 0] Double) H.=== [],
      testProperty "position offsets preserve chunk equivalence and embedding gradients" $ H.withTests 1 $ H.property $ do
        let first = NN.sinusoidalPositions @2 @4 0 :: T.Tensor '[2, 4] Double
            second = NN.sinusoidalPositions @2 @4 2 :: T.Tensor '[2, 4] Double
            full = NN.sinusoidalPositions @4 @4 0 :: T.Tensor '[4, 4] Double
        approxList (values first ++ values second) $ values full
        let inputs = T.ones @'[2, 3, 4] @Double
            (_, gradient) = AG.backprop (\x -> x + T.broadcast (NN.sinusoidalPositions @3 @4 0)) inputs
        values gradient H.=== replicate 24 1
        approxList (map realToFrac $ T.toList $ T.run T.CPU (NN.sinusoidalPositions @2 @3 2 :: T.Tensor '[2, 3] Float)) $ values $ NN.sinusoidalPositions @2 @3 2,
      testProperty "learned positions use embedding with ordinary gradient accumulation" $ H.withTests 1 $ H.property $ do
        let table = NN.Embedding $ T.zeros @'[4, 2] @Double
            (_, gradient) = AG.backprop (\p -> NN.embedding p $ T.fromList @'[3] [1, 2, 1]) table
        values (NN.embeddingWeights gradient) H.=== [0, 0, 2, 2, 1, 1, 0, 0],
      testProperty "RoPE agrees with independent pair rotations across batches and heads" $ H.withTests 1 $ H.property $ do
        let inputs = [sin (fromIntegral index) | index <- [1 .. 48 :: Int]]
            sequences [] = []
            sequences xs = let (first, rest) = splitAt 12 xs in rows first : sequences rest
            rows [] = []
            rows xs = let (first, rest) = splitAt 4 xs in first : rows rest
            expected = concatMap (concat . rotate 1000 5) $ sequences inputs
            actual = NN.rope 1000 5 $ T.fromList @'[2, 2, 3, 4] inputs
        approxList (values actual) expected,
      testProperty "RoPE pullback agrees with finite differences and nonuniform seeds" $ H.withTests 1 $ H.property $ do
        let inputs = [0.2, -0.4, 0.7, 0.3, 0.5, 0.6, -0.2, 0.1]
            seed = [0.3, -0.4, 0.5, 0.8, -0.2, 0.6, 0.9, -0.7]
            (_, backward) = AG.pullback (NN.rope 10000 3) $ T.fromList @'[1, 1, 2, 4] inputs
            gradient = backward $ T.fromList seed
            objective xs = sum $ zipWith (*) seed $ concat $ rotate 10000 3 [take 4 xs, drop 4 xs]
            epsilon = 1e-5
            change index delta = [if i == index then x + delta else x | (i, x) <- zip [0 :: Int ..] inputs]
            expected = [(objective (change index epsilon) - objective (change index (-epsilon))) / (2 * epsilon) | index <- [0 .. 7]]
        approxList (values gradient) expected,
      testProperty "RoPE preserves norms and position zero is the identity" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[1, 1, 2, 4] [1, 2, 3, 4, -2, 1, 0.5, 0.3 :: Double]
            rotated = NN.rope 10000 7 input
            zero = T.fromList @'[1, 1, 1, 4] [1, 2, 3, 4 :: Double]
        approxList (values $ T.sumAxes @'[3] $ rotated * rotated) $ values $ T.sumAxes @'[3] $ input * input
        approxList (values $ NN.rope 10000 0 zero) $ values zero,
      testProperty "RoPE offset makes chunked and whole sequence processing agree" $ H.withTests 1 $ H.property $ do
        let inputs = [0.1 * fromIntegral i | i <- [1 .. 16 :: Int]]
            whole = NN.rope 500 5 $ T.fromList @'[1, 1, 4, 4] inputs
            first = NN.rope 500 5 $ T.fromList @'[1, 1, 2, 4] $ take 8 inputs
            second = NN.rope 500 7 $ T.fromList @'[1, 1, 2, 4] $ drop 8 inputs
        approxList (values $ T.concat @2 first second) $ values whole,
      testProperty "RoPE attention depends on relative positions and differentiates Q and K" $ H.withTests 1 $ H.property $ do
        let q = T.fromList @'[1, 1, 2, 2] [0.2, -0.4, 0.7, 0.3 :: Double]
            k = T.fromList @'[1, 1, 3, 2] [0.5, 0.6, -0.2, 0.1, 0.3, -0.7]
            v = T.fromList @'[1, 1, 3, 1] [1, 3, -2]
            result offset = AG.backprop (\(query, key) -> NN.attention Nothing (NN.rope 10000 offset query) (NN.rope 10000 offset key) (AG.constant v)) (q, k)
            (first, (dq, dk)) = result 2
            (shifted, (shiftedDQ, shiftedDK)) = result 12
        approxList (values first) $ values shifted
        approxList (values dq) $ values shiftedDQ
        approxList (values dk) $ values shiftedDK
        H.assert $ any ((> 1e-6) . abs) $ values dq
        H.assert $ any ((> 1e-6) . abs) $ values dk,
      testProperty "RoPE supports Float changing prepared inputs and Forward" $ H.withTests 1 $ H.property $ do
        let execute = T.run1 T.CPU $ NN.rope 10000 4
            input = T.fromListData @'[1, 1, 2, 2] [1, 2, 3, 4 :: Float]
            doubled = T.fromListData @'[1, 1, 2, 2] [2, 4, 6, 8 :: Float]
            expected = concat $ rotate 10000 4 [[1, 2], [3, 4]]
        approxList (map realToFrac $ T.toList $ execute input) expected
        approxList (map realToFrac $ T.toList $ execute doubled) $ map (* 2) expected
        let (output, state) = NN.runState (NN.forward (NN.rope 10000 4) $ T.fromData input) True
        state H.=== True
        approxList (T.toList $ T.run T.CPU output) $ T.toList $ execute input,
      testProperty "RoPE preserves empty dimensions and their gradients" $ H.withTests 1 $ H.property $ do
        let empty = T.zeros @'[1, 2, 0, 4] @Double
            (output, gradient) = AG.backprop (NN.rope 10000 5) empty
        values output H.=== []
        values gradient H.=== []
        values (NN.rope 10000 0 (T.zeros @'[0, 1, 2, 4] @Double)) H.=== []
        values (NN.rope 10000 0 (T.zeros @'[1, 0, 2, 4] @Double)) H.=== []
        let (zeroWidth, zeroGradient) = AG.backprop (NN.rope 10000 0) $ T.zeros @'[1, 1, 2, 0] @Double
        values zeroWidth H.=== []
        values zeroGradient H.=== [],
      testProperty "RoPE rejects odd feature widths" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/RoPEWidth.hs" "Couldn't match type"
    ]
