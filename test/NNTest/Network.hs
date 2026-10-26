{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module NNTest.Network (tests) where

import Control.Arrow
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Ops.Elementwise as E
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Parameters t a = Parameters
  { hiddenLayer :: NN.Dense 2 2 t a,
    outputLayer :: NN.Dense 2 1 t a
  }
  deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor Double) = Parameters t Double

newtype Shared t a = Shared {sharedLayer :: NN.Dense 2 2 t a}
  deriving (Generic)

type instance T.WithTensor t (Shared T.Tensor Double) = Shared t Double

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

parameters :: Parameters T.Tensor Double
parameters =
  Parameters
    (NN.Dense (T.fromList @'[2, 2] [1, -1, 2, 1]) (T.zeros @'[2]))
    (NN.Dense (T.fromList @'[2, 1] [3, 4]) (T.fromList @'[1] [0.5]))

network ::
  ( T.TensorOps t a,
    Num (T.TensorElement a),
    Num (t '[2] a),
    Num (t '[1] a),
    E.Ord (t '[2] a),
    E.Conditional (t '[2] a),
    E.Condition (t '[2] a) ~ E.Comparison (t '[2] a)
  ) =>
  Parameters t a ->
  t '[2] a ->
  t '[1] a
network p = proc input -> do
  hidden <- NN.dense (hiddenLayer p) -< input
  NN.dense (outputLayer p) -< NN.relu hidden

sharedNetwork ::
  (T.TensorOps t a, Num (T.TensorElement a), Num (t '[2] a)) =>
  Shared t a ->
  (t '[2] a, t '[2] a) ->
  (t '[2] a, t '[2] a)
sharedNetwork p = proc (firstInput, secondInput) -> do
  leftOutput <- NN.dense (sharedLayer p) -< firstInput
  rightOutput <- NN.dense (sharedLayer p) -< secondInput
  returnA -< (leftOutput, rightOutput)

tests :: TestTree
tests =
  testGroup
    "network"
    [ testProperty "inference needs no Differentiable instance for the parameter record" $ H.withTests 1 $ H.property $ do
        let p =
              Parameters
                (NN.Dense (T.fromList @'[2, 2] [1, -1, 2, 1]) (T.zeros @'[2]))
                (NN.Dense (T.fromList @'[2, 1] [3, 4]) (T.fromList @'[1] [0.5])) ::
                Parameters T.Tensor Float
        T.toList (T.run T.CPU (network p (T.fromList @'[2] [1, 2]))) H.=== [19.5],
      testProperty "arrows and ordinary function composition use the same parameters" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [1, 2 :: Double]
            composed = NN.dense (hiddenLayer parameters) >>> NN.relu >>> NN.dense (outputLayer parameters)
            evaluated = AG.evaluate (\p -> network p (AG.constant input)) parameters
        values (network parameters input) H.=== [19.5]
        values (composed input) H.=== values evaluated,
      testProperty "one network function differentiates its parameter record and input" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [1, 2 :: Double]
            (loss, (gradient, dx)) = AG.backprop (\(p, x) -> T.sum (network p x)) (parameters, input)
        values loss H.=== [19.5]
        approxList (values (NN.denseWeights (hiddenLayer gradient))) [3, 4, 6, 8]
        approxList (values (NN.denseBias (hiddenLayer gradient))) [3, 4]
        approxList (values (NN.denseWeights (outputLayer gradient))) [5, 1]
        approxList (values (NN.denseBias (outputLayer gradient))) [1]
        approxList (values dx) [-1, 10],
      testProperty "shared fields accumulate gradients across multiple inputs and outputs" $ H.withTests 1 $ H.property $ do
        let p = Shared (NN.Dense (T.fromList @'[2, 2] [1, 0, 0, 1]) (T.zeros @'[2])) :: Shared T.Tensor Double
            firstInput = T.fromList @'[2] [1, 2 :: Double]
            secondInput = T.fromList @'[2] [3, 4 :: Double]
            ((leftOutput, rightOutput), gradient) = AG.backprop (\s -> sharedNetwork s (AG.constant firstInput, AG.constant secondInput)) p
        values leftOutput H.=== [1, 2]
        values rightOutput H.=== [3, 4]
        values (NN.denseWeights (sharedLayer gradient)) H.=== [4, 4, 6, 6]
        values (NN.denseBias (sharedLayer gradient)) H.=== [2, 2]
    ]
