{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Reduction.Masked (tests) where

import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh, T.DType a) => T.Tensor sh a -> [a]
values = T.toList . T.run T.CPU

input :: T.Tensor '[2, 3] Double
input = T.fromList @'[2, 3] [1, 2, 3, 4, 5, 6]

mask :: T.Tensor '[2, 3] Bool
mask = T.fromList @'[2, 3] [True, False, True, True, True, False]

tests :: TestTree
tests =
    testGroup
        "masked reductions"
        [ testProperty "statistics agree with selected values" $ H.withTests 20 $ H.property $ do
            xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.double $ Range.linearFrac (-10) 10)
            flags <- H.forAll $ Gen.list (Range.singleton 6) Gen.bool
            let selected = [x | (x, True) <- zip xs flags]
                tensor = T.fromList @'[6] xs
                chosen = T.fromList @'[6] flags
                count = fromIntegral $ length selected
                avg = sum selected / count
                var = sum [(x - avg) * (x - avg) | x <- selected] / count
            if null selected
                then pure ()
                else do
                    approxList (values $ T.maskedMean chosen tensor) [avg]
                    approxList (values $ T.maskedVariance 0 chosen tensor) [var]
                    approxList (values $ T.maskedStddev 0 chosen tensor) [sqrt var]
        , testProperty "ordinary and masked statistics reduce axes and support correction" $ H.withTests 1 $ H.property $ do
            approxList (values $ T.variance 0 input) [35 / 12]
            approxList (values $ T.varianceAxes @'[1] 1 input) [1, 1]
            approxList (values $ T.stddevAxes @'[0] 0 input) [1.5, 1.5, 1.5]
            approxList (values $ T.maskedMeanAxes @'[1] mask input) [2, 4.5]
            approxList (values $ T.maskedVarianceAxes @'[1] 1 mask input) [2, 0.5]
            approxList (values $ T.maskedStddev 0 mask input) [sqrt 2.1875]
        , testProperty "masked statistics and gradients ignore masked NaNs" $ H.withTests 1 $ H.property $ do
            let source = T.fromList @'[3] [1, 0 / 0, 3 :: Double]
                selected = T.fromList @'[3] [True, False, True]
            approxList (values $ AG.gradient (T.maskedMean selected) source) [0.5, 0, 0.5]
            approxList (values $ AG.gradient (T.maskedVariance 0 selected) source) [-1, 0, 1]
            approxList (values $ AG.gradient (T.maskedStddev 0 selected) source) [-0.5, 0, 0.5]
        , testProperty "extrema use the supplied fill value" $ H.withTests 1 $ H.property $ do
            values (T.maskedMinimumAxes @'[1] 99 mask input) H.=== [1, 4]
            values (T.maskedMaximumAxes @'[1] (-99) mask input) H.=== [3, 5]
            values (T.maskedArgmin @1 99 mask input) H.=== [0, 0]
            values (T.maskedArgmax @1 (-99) mask input) H.=== [2, 1]
            values (T.maskedMinimum 0 mask input) H.=== [0]
            values (T.maskedArgmax @1 100 mask input) H.=== [1, 2]
        , testProperty "masked sum, booleans and norms" $ H.withTests 1 $ H.property $ do
            values (T.maskedSum mask input) H.=== [13]
            values (T.maskedSumAxes @'[1] mask input) H.=== [4, 9]
            values (T.maskedAllAxes @'[1] mask mask) H.=== [True, True]
            values (T.maskedAnyAxes @'[1] mask mask) H.=== [True, True]
            approxList (values $ T.maskedNormL1 mask input) [13]
            approxList (values $ T.maskedNormL2 mask input) [sqrt 51]
            approxList (values $ AG.gradient (T.maskedSum mask) input) [1, 0, 1, 1, 1, 0]
        , testProperty "masked normalization handles finite selected logits" $ H.withTests 1 $ H.property $ do
            let source = T.fromList @'[3] [1000, 0 / 0, 1000 :: Double]
                selected = T.fromList @'[3] [True, False, True]
            approxList (values $ T.maskedSoftmax @0 selected source) [0.5, 0, 0.5]
            approxList (values $ T.maskedLogSumExp selected source) [1000 + log 2]
            approxList (values $ AG.gradient (T.maskedLogSumExp selected) source) [0.5, 0, 0.5]
        , testProperty "expanding reduced axes and their gradients" $ H.withTests 1 $ H.property $ do
            let reduced = T.fromList @'[2] [10, 20 :: Double]
            values (T.expandReduced @'[1] input reduced) H.=== [10, 10, 10, 20, 20, 20]
            approxList (values $ AG.gradient (\x -> T.sum $ T.expandReduced @'[1] (AG.constant input) x) reduced) [3, 3]
        ]
