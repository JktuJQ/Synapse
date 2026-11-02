{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module NNTest.Optimizer (tests) where

import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.NN.Optimizers as O
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Parameters t a = Parameters {projection :: NN.Dense 1 1 t a, scale :: t '[] a} deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor a) = Parameters t a

newtype Ascent = Ascent Double

instance (AG.Differentiable p, AG.ParametersConstraints Fractional p) => O.Optimizer Ascent p where
  type OptimizerState Ascent p = ()
  initialState _ _ = ()
  optimizerStep (Ascent lr) parameters gradients () =
    (O.updateParameters @Fractional (\p g -> p + realToFrac lr * g) parameters gradients, ())

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "optimizers"
    [ testProperty "momentum and Nesterov follow successive analytic updates" $ H.withTests 1 $ H.property $ do
        let run nesterov =
              let optimizer = NN.Momentum 0.1 0.9 nesterov
                  p = T.full @'[] (5 :: Double)
                  (p1, v1) = NN.optimizerStep optimizer p 2 (NN.initialState optimizer p)
                  (p2, v2) = NN.optimizerStep optimizer p1 3 v1
               in (values p1, values p2, values v2)
            (first, second, velocity) = run False
            (firstN, secondN, _) = run True
        approxList first [4.8]
        approxList second [4.32]
        approxList velocity [4.8]
        approxList firstN [4.62]
        approxList secondN [3.888],
      testProperty "Adam and AdamW correct both moments over multiple steps" $ H.withTests 1 $ H.property $ do
        let adam = NN.Adam 0.1 0.9 0.99 1e-8
            p = T.full @'[] (5 :: Double)
            (p1, state1) = NN.optimizerStep adam p 2 (NN.initialState adam p)
            (p2, (count, (m, v))) = NN.optimizerStep adam p1 3 state1
            m2 = 0.9 * 0.2 + 0.1 * 3
            v2 = 0.99 * 0.04 + 0.01 * 9
            expected1 = 5 - 0.1 * 2 / (2 + 1e-8)
            expected2 = expected1 - 0.1 * (m2 / (1 - 0.9 ** 2)) / (sqrt (v2 / (1 - 0.99 ** 2)) + 1e-8)
            adamW = NN.AdamW adam 0.2
            (decayed, _) = NN.optimizerStep adamW p 2 (NN.initialState adamW p)
        T.toList (T.run T.CPU count) H.=== [2]
        approxList (values p1) [expected1]
        approxList (values p2) [expected2]
        approxList (values m) [m2]
        approxList (values v) [v2]
        approxList (values decayed) [expected1 - 0.1],
      testProperty "RMSprop accumulates squared gradients across prepared calls" $ H.withTests 1 $ H.property $ do
        let optimizer = NN.RMSprop 0.1 0.9 1e-8
            execute = T.run3 T.CPU $ \parameters gradient state -> NN.optimizerStep optimizer parameters gradient state
            p0 = T.run T.CPU $ T.full @'[] (5 :: Double)
            g = T.run T.CPU $ T.full @'[] (2 :: Double)
            v0 = T.run T.CPU $ NN.initialState optimizer $ T.full @'[] (5 :: Double)
            (p1, v1) = execute p0 g v0
            (p2, v2) = execute p1 g v1
        approxList (T.toList p2) [5 - 0.2 / (sqrt 0.4 + 1e-8) - 0.2 / (sqrt 0.76 + 1e-8)]
        approxList (T.toList v2) [0.76],
      testProperty "Adam state passes through repeated prepared backend calls" $ H.withTests 1 $ H.property $ do
        let optimizer = NN.Adam 0.1 0.9 0.999 1e-8
            execute = T.run3 T.CPU $ \parameters gradient state -> NN.optimizerStep optimizer parameters gradient state
            p = T.full @'[] (5 :: Double)
            p0 = T.run T.CPU p
            g = T.run T.CPU $ T.full @'[] (2 :: Double)
            state0 = T.run T.CPU $ NN.initialState optimizer p
            (p1, state1) = execute p0 g state0
            (p2, state2) = execute p1 g state1
        approxList (T.toList p2) [5 - 0.4 / (2 + 1e-8)]
        T.toList (fst state2) H.=== [2],
      testProperty "Adam updates nested tensor records without numeric record instances" $ H.withTests 1 $ H.property $ do
        let p = Parameters (NN.Dense 2 3) 4 :: Parameters T.Tensor Double
            g = Parameters (NN.Dense 1 2) 3
            optimizer = NN.Adam 0.1 0.9 0.999 1e-8
            (updated, _) = NN.optimizerStep optimizer p g (NN.initialState optimizer p)
        approxList (values $ NN.denseWeights $ projection updated) [2 - 0.1 / (1 + 1e-8)]
        approxList (values $ NN.denseBias $ projection updated) [3 - 0.2 / (2 + 1e-8)]
        approxList (values $ scale updated) [4 - 0.3 / (3 + 1e-8)],
      testProperty "updateParameters passes parameters before gradients" $ H.withTests 1 $ H.property $ do
        let parameters :: Parameters T.Tensor Double
            parameters = Parameters (NN.Dense 6 12) 20
            gradients = Parameters (NN.Dense 2 4) 5
            updated = NN.updateParameters @Fractional (/) parameters gradients
        values (NN.denseWeights (projection updated)) H.=== [3]
        values (NN.denseBias (projection updated)) H.=== [3]
        values (scale updated) H.=== [4],
      testProperty "updateParameters supplies Floating operations to every tensor" $ H.withTests 1 $ H.property $ do
        let parameters :: Parameters T.Tensor Double
            parameters = Parameters (NN.Dense 8 12) 20
            gradients = Parameters (NN.Dense 4 9) 16
            updated = NN.updateParameters @Floating (\p g -> p - sqrt g) parameters gradients
        values (NN.denseWeights (projection updated)) H.=== [6]
        values (NN.denseBias (projection updated)) H.=== [9]
        values (scale updated) H.=== [16],
      testProperty "custom optimizers use only public parameter traversal" $ H.withTests 1 $ H.property $ do
        let parameters :: Parameters T.Tensor Double
            parameters = Parameters (NN.Dense 1 2) 3
            gradients = Parameters (NN.Dense 2 4) 6
            optimizer = Ascent 0.1
            (updated, ()) = NN.optimizerStep optimizer parameters gradients (NN.initialState optimizer parameters)
        approxList (values (NN.denseWeights (projection updated))) [1.2]
        approxList (values (NN.denseBias (projection updated))) [2.4]
        approxList (values (scale updated)) [3.6],
      testProperty "SGD updates heterogeneous nested tensor parameters" $ H.withTests 1 $ H.property $ do
        let parameters :: Parameters T.Tensor Double
            parameters = Parameters (NN.Dense 2 3) 4
            gradients = Parameters (NN.Dense 1 2) 3
            config = NN.SGD 0.1
            (updated, state) = NN.optimizerStep config parameters gradients (NN.initialState config parameters)
        state H.=== ()
        approxList (values (NN.denseWeights (projection updated))) [1.9]
        approxList (values (NN.denseBias (projection updated))) [2.8]
        approxList (values (scale updated)) [3.7],
      testProperty "SGD preserves Float tensors and empty tensors" $ H.withTests 1 $ H.property $ do
        let (tensor, ()) = NN.optimizerStep (NN.SGD 0.5) (T.full @'[2] (3 :: Float)) 2 ()
            (empty, ()) = NN.optimizerStep (NN.SGD 0.1) (T.zeros @'[0] @Double) 1 ()
        T.toList (T.run T.CPU tensor) H.=== [2, 2]
        values empty H.=== [],
      testProperty "SGD applies gradients of a dense regression objective" $ H.withTests 1 $ H.property $ do
        let parameters :: Parameters T.Tensor Double
            parameters = Parameters (NN.Dense 0 0) 1
            input = T.fromList @'[1] [2 :: Double]
            target = T.full @'[1] (4 :: Double)
            objective :: (AG.Context s) => Parameters (AG.Tensor s) Double -> AG.Tensor s '[] Double
            objective p = NN.mse T.sum (NN.dense (projection p) (AG.constant input) * T.broadcast (scale p)) (AG.constant target)
            (loss, gradients) = AG.backprop objective parameters
            (updated, ()) = NN.optimizerStep (NN.SGD 0.1) parameters gradients ()
        values loss H.=== [16]
        values (NN.denseWeights (projection gradients)) H.=== [-16]
        values (NN.denseBias (projection gradients)) H.=== [-8]
        values (scale gradients) H.=== [0]
        approxList (values (AG.evaluate objective updated)) [0],
      testProperty "zero learning rates preserve tensor values" $ H.withTests 1 $ H.property $ do
        let (unchanged, ()) = NN.optimizerStep (NN.SGD 0) (T.full @'[] (3 :: Double)) 2 ()
            (updated, ()) = NN.optimizerStep (NN.SGD 0.2) unchanged 2 ()
        values unchanged H.=== [3]
        approxList (values updated) [2.6],
      testProperty "SGD repeatedly trains a prepared scalar tensor objective" $ H.withTests 1 $ H.property $ do
        let function :: T.Tensor '[] Double -> T.Tensor '[] Double
            function p =
              let g = AG.gradient (\x -> (x - 3) * (x - 3)) p
               in fst (NN.optimizerStep (NN.SGD 0.25) p g ())
            execute = T.run1 T.CPU function
            initial = T.run T.CPU (T.full @'[] (0 :: Double))
        approxList (T.toList (iterate execute initial !! 12)) [3 * (1 - 0.5 ** 12)],
      testProperty "SGD rejects integral tensor parameters" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/SGDInt.hs" "Fractional"
    ]
