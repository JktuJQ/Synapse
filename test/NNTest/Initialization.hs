module NNTest.Initialization (tests) where

import Control.Exception (ErrorCall, evaluate, try)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import System.Random (StdGen, mkStdGen64, uniformRs)
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: T.Shape sh => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.defaultCPU

seed :: StdGen
seed = mkStdGen64 2026

rejects :: T.Tensor '[1] Double -> H.PropertyT IO ()
rejects tensor = do
    result <- H.evalIO $ try @ErrorCall $ evaluate tensor
    case result of
        Left _ -> pure ()
        Right _ -> H.failure

moments :: [Double] -> (Double, Double)
moments xs =
    let count = fromIntegral (length xs)
        mean = sum xs / count
     in (mean, sum (map (\x -> (x - mean) * (x - mean)) xs) / count)

tests :: TestTree
tests =
    testGroup
        "initialization"
        [ testProperty "uniform fills the requested shape using the supplied generator" $ H.withTests 1 $ H.property $ do
            let actual = values (NN.evalState (NN.uniform @'[2, 3] (-2) 4) seed)
            actual H.=== take 6 (uniformRs (-2, 4) seed)
            H.assert (all (\x -> x >= -2 && x <= 4) actual)
        , testProperty "sequential initialization is reproducible and advances the generator" $ H.withTests 1 $ H.property $ do
            let initialize = do
                    left <- NN.normal @'[32] 0 1
                    right <- NN.normal @'[32] 0 1
                    pure (left, right)
                ((first, second), next) = NN.runState initialize seed
                ((againFirst, againSecond), againNext) = NN.runState initialize seed
            values first H.=== values againFirst
            values second H.=== values againSecond
            next H.=== againNext
            H.assert (next /= seed)
            H.assert (values first /= values second)
        , testProperty "uniform initialization resumes exactly at the returned generator" $ H.withTests 1 $ H.property $ do
            let (first, next) = NN.runState (NN.uniform @'[3] 0 1) seed
                (second, final) = NN.runState (NN.uniform @'[5] 0 1) next
                (whole, expectedFinal) = NN.runState (NN.uniform @'[8] 0 1) seed
            values first ++ values second H.=== values whole
            final H.=== expectedFinal
        , testProperty "normal initialization resumes after complete sample pairs" $ H.withTests 1 $ H.property $ do
            let (first, next) = NN.runState (NN.normal @'[4] 0 1) seed
                (second, final) = NN.runState (NN.normal @'[6] 0 1) next
                (whole, expectedFinal) = NN.runState (NN.normal @'[10] 0 1) seed
            values first ++ values second H.=== values whole
            final H.=== expectedFinal
        , testProperty "an odd normal sample count consumes the complete final pair" $ H.withTests 1 $ H.property $ do
            let (one, nextOne) = NN.runState (NN.normal @'[1] 0 1) seed
                (pair, nextPair) = NN.runState (NN.normal @'[2] 0 1) seed
            values one H.=== take 1 (values pair)
            nextOne H.=== nextPair
        , testProperty "empty and constant initializers preserve generator state" $ H.withTests 1 $ H.property $ do
            let initialize = do
                    _ <- NN.uniform @'[0] @Double 0 1
                    _ <- NN.normal @'[0] @Double 0 1
                    _ <- NN.uniform @'[3] @Double 2 2
                    _ <- NN.normal @'[3] @Double 2 0
                    _ <- NN.xavierUniform @'[3] @Double 2 3 0
                    _ <- NN.xavierNormal @'[3] @Double 2 3 0
                    pure ()
            snd (NN.runState initialize seed) H.=== seed
        , testProperty "normal samples have the requested mean and variance" $ H.withTests 1 $ H.property $ do
            let samples = values (NN.evalState (NN.normal @'[50000] 2 3) seed)
                (mean, variance) = moments samples
            H.assert (all (\x -> not (isNaN x || isInfinite x)) samples)
            H.annotateShow (mean, variance)
            H.assert (abs (mean - 2) < 0.06)
            H.assert (abs (variance - 9) < 0.25)
        , testProperty "uniform samples have the requested mean and variance" $ H.withTests 1 $ H.property $ do
            let (mean, variance) = moments (values (NN.evalState (NN.uniform @'[50000] (-2) 4) seed))
            H.annotateShow (mean, variance)
            H.assert (abs (mean - 1) < 0.04)
            H.assert (abs (variance - 3) < 0.08)
        , testProperty "xavier uses explicit fans and gain independently of result shape" $ H.withTests 1 $ H.property $ do
            let fanIn = 7
                fanOut = 11
                gain = 1.5 :: Double
                std = gain * sqrt (2 / 18)
                bound = gain * sqrt (6 / 18)
            approxList
                (values (NN.evalState (NN.xavierUniform @'[3, 5] fanIn fanOut gain) seed))
                (values (NN.evalState (NN.uniform @'[3, 5] (-bound) bound) seed))
            approxList
                (values (NN.evalState (NN.xavierNormal @'[3, 5] fanIn fanOut gain) seed))
                (values (NN.evalState (NN.normal @'[3, 5] 0 std) seed))
        , testProperty "kaiming supports relu and leaky relu with an explicitly selected fan" $ H.withTests 1 $ H.property $ do
            let fan = 7
                slope = 0.2 :: Double
                std = sqrt (2 / ((1 + slope * slope) * 7))
                bound = sqrt (6 / ((1 + slope * slope) * 7))
            approxList
                (values (NN.evalState (NN.kaimingUniform @'[15] fan slope) seed))
                (values (NN.evalState (NN.uniform @'[15] (-bound) bound) seed))
            approxList
                (values (NN.evalState (NN.kaimingNormal @'[15] fan slope) seed))
                (values (NN.evalState (NN.normal @'[15] 0 std) seed))
            approxList
                (values (NN.evalState (NN.kaimingNormal @'[15] fan 0) seed))
                (values (NN.evalState (NN.normal @'[15] 0 (sqrt (2 / 7))) seed))
        , testProperty "degenerate distributions and zero gain produce constants" $ H.withTests 1 $ H.property $ do
            values (NN.evalState (NN.uniform @'[3] 4 4) seed) H.=== [4, 4, 4]
            values (NN.evalState (NN.normal @'[3] 4 0) seed) H.=== [4, 4, 4]
            values (NN.evalState (NN.xavierUniform @'[3] 2 3 0) seed) H.=== [0, 0, 0]
            values (NN.evalState (NN.xavierNormal @'[3] 2 3 0) seed) H.=== [0, 0, 0]
        , testProperty "initializers support scalars, empty tensors, and odd sample counts" $ H.withTests 1 $ H.property $ do
            length (values (NN.evalState (NN.normal @'[] 0 1) seed)) H.=== 1
            length (values (NN.evalState (NN.normal @'[5] 0 1) seed)) H.=== 5
            values (NN.evalState (NN.uniform @'[0, 3] (-1) 1) seed) H.=== []
            values (NN.evalState (NN.normal @'[0, 3] 0 1) seed) H.=== []
            values (NN.evalState (NN.xavierUniform @'[0] 2 3 1) seed) H.=== []
            values (NN.evalState (NN.xavierNormal @'[0] 2 3 1) seed) H.=== []
            values (NN.evalState (NN.kaimingUniform @'[0] 2 0) seed) H.=== []
            values (NN.evalState (NN.kaimingNormal @'[0] 2 0) seed) H.=== []
        , testProperty "initializers preserve Float precision and support all initialization variants" $ H.withTests 1 $ H.property $ do
            let runFloat = T.toList . T.run T.defaultCPU :: T.Tensor '[8] Float -> [Float]
                finite xs = length xs == 8 && all (\x -> not (isNaN x || isInfinite x)) xs
            H.assert (finite (runFloat (NN.evalState (NN.uniform (-1) 1) seed)))
            runFloat (NN.evalState (NN.normal 0 1) seed)
                H.=== map realToFrac (values (NN.evalState (NN.normal @'[8] 0 1) seed))
            H.assert (finite (runFloat (NN.evalState (NN.xavierUniform 2 4 1) seed)))
            H.assert (finite (runFloat (NN.evalState (NN.xavierNormal 2 4 1) seed)))
            H.assert (finite (runFloat (NN.evalState (NN.kaimingUniform 2 0.1) seed)))
            H.assert (finite (runFloat (NN.evalState (NN.kaimingNormal 2 0.1) seed)))
        , testProperty "initialized tensors can be registered for differentiation" $ H.withTests 1 $ H.property $ do
            let parameters = NN.evalState (NN.xavierNormal @'[2, 3] 2 3 1) seed :: T.Tensor '[2, 3] Double
                (_, gradient) = AG.backprop (T.sum . (\x -> x * x)) parameters
            approxList (values gradient) (map (* 2) (values parameters))
        , testProperty "fan sums and large finite slopes do not overflow intermediate calculations" $ H.withTests 1 $ H.property $ do
            let largeFan = maxBound :: Int
                samples = values (NN.evalState (NN.xavierNormal @'[3] largeFan largeFan 1) seed)
                tinySamples = values (NN.evalState (NN.kaimingNormal @'[3] 1 1e200) seed)
            H.assert (all (\x -> not (isNaN x || isInfinite x)) samples)
            H.assert (any (/= 0) samples)
            H.assert (all (\x -> not (isNaN x || isInfinite x)) tinySamples)
            H.assert (any (/= 0) tinySamples)
        , testProperty "invalid distribution parameters are rejected" $ H.withTests 1 $ H.property $ do
            rejects (NN.evalState (NN.uniform @'[1] 2 1) seed)
            mapM_ (\x -> rejects (NN.evalState (NN.uniform @'[1] x 1) seed)) [0 / 0, 1 / 0, -(1 / 0)]
            mapM_ (\x -> rejects (NN.evalState (NN.uniform @'[1] 0 x) seed)) [0 / 0, 1 / 0, -(1 / 0)]
            mapM_ (\x -> rejects (NN.evalState (NN.normal @'[1] x 1) seed)) [0 / 0, 1 / 0, -(1 / 0)]
            mapM_ (\x -> rejects (NN.evalState (NN.normal @'[1] 0 x) seed)) [-1, 0 / 0, 1 / 0, -(1 / 0)]
        , testProperty "invalid fans, gains, and slopes are rejected" $ H.withTests 1 $ H.property $ do
            mapM_ (\fan -> rejects (NN.evalState (NN.xavierUniform @'[1] fan 2 1) seed)) [0, -1]
            mapM_ (\fan -> rejects (NN.evalState (NN.xavierNormal @'[1] 2 fan 1) seed)) [0, -1]
            mapM_ (\gain -> rejects (NN.evalState (NN.xavierUniform @'[1] 2 3 gain) seed)) [-1, 0 / 0, 1 / 0, -(1 / 0)]
            mapM_ (\gain -> rejects (NN.evalState (NN.xavierNormal @'[1] 2 3 gain) seed)) [-1, 0 / 0, 1 / 0, -(1 / 0)]
            mapM_ (\fan -> rejects (NN.evalState (NN.kaimingUniform @'[1] fan 0) seed)) [0, -1]
            mapM_ (\fan -> rejects (NN.evalState (NN.kaimingNormal @'[1] fan 0) seed)) [0, -1]
            mapM_ (\slope -> rejects (NN.evalState (NN.kaimingUniform @'[1] 2 slope) seed)) [0 / 0, 1 / 0, -(1 / 0)]
            mapM_ (\slope -> rejects (NN.evalState (NN.kaimingNormal @'[1] 2 slope) seed)) [0 / 0, 1 / 0, -(1 / 0)]
        ]
