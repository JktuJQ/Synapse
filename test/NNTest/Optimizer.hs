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
import qualified Synapse.Autograd.Differentiable as D
import qualified Synapse.NN as NN
import qualified Synapse.NN.Optimizer as O
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Parameters t a = Parameters
  { projection :: NN.Dense 1 1 t a,
    scale :: t '[] a
  }
  deriving (Generic)

instance AG.Differentiable (Parameters T.Tensor Double) where
  type WithGrad s (Parameters T.Tensor Double) = Parameters (AG.Tensor s) Double

data Manual = Manual
  { weights :: T.Tensor '[2] Double,
    bias :: Double
  }

data ManualVariables s = ManualVariables
  { weightVariables :: AG.Tensor s '[2] Double,
    biasVariable :: AG.Var s Double
  }

instance AG.Differentiable Manual where
  type WithGrad s Manual = ManualVariables s
  type ParametersConstraints c Manual = D.FieldConstraints c '[T.Tensor '[2] Double, Double]
  structureWith =
    D.record Manual ManualVariables
      D.<+> D.field weights weightVariables
      D.<+> D.field bias biasVariable

data GenericManual = GenericManual
  { genericWeights :: T.Tensor '[2] Double,
    genericBias :: Double
  }
  deriving (Generic)

data GenericVariables s = GenericVariables
  { genericWeightVariables :: AG.Tensor s '[2] Double,
    genericBiasVariable :: AG.Var s Double
  }

instance AG.Differentiable GenericManual where
  type WithGrad s GenericManual = GenericVariables s
  structureWith =
    D.record GenericManual GenericVariables
      D.<+> D.field genericWeights genericWeightVariables
      D.<+> D.field genericBias genericBiasVariable

newtype Ascent = Ascent Double

instance (AG.Differentiable p, D.ParametersConstraints Fractional p) => O.Optimizer Ascent p where
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
    [ testProperty "updateParameters passes parameters before gradients for both structures and formulas" $ H.withTests 1 $ H.property $ do
        let parameters = GenericManual (T.fromList @'[2] [6, 12]) 20
            gradients = GenericManual (T.fromList @'[2] [2, 4]) 5
            updated = NN.updateParameters @Fractional (/) parameters gradients
        values (genericWeights updated) H.=== [3, 3]
        genericBias updated H.=== 4,
      testProperty "updateParameters accepts integral leaves with Num requirements" $ H.withTests 1 $ H.property $ do
        let parameters = (10 :: Int, 20 :: Integer)
            gradients = (3, 4)
            updated = NN.updateParameters @Num (\p g -> p - 2 * g) parameters gradients
        updated H.=== (4, 12),
      testProperty "updateParameters supplies Floating operations to each component" $ H.withTests 1 $ H.property $ do
        let parameters = GenericManual (T.fromList @'[2] [8, 12]) 20
            gradients = GenericManual (T.fromList @'[2] [4, 9]) 16
            updated = NN.updateParameters @Floating (\p g -> p - sqrt g) parameters gradients
        values (genericWeights updated) H.=== [6, 9]
        genericBias updated H.=== 16,
      testProperty "custom optimizers can update records using only public APIs" $ H.withTests 1 $ H.property $ do
        let parameters = GenericManual (T.fromList @'[2] [1, 2]) 3
            gradients = GenericManual (T.fromList @'[2] [2, 4]) 6
            optimizer = Ascent 0.1
            (updated, ()) = O.optimizerStep optimizer parameters gradients (O.initialState optimizer parameters)
        approxList (values (genericWeights updated)) [1.2, 2.4]
        approxList [genericBias updated] [3.6],
      testProperty "SGD updates heterogeneous nested parameters without numeric record instances" $ H.withTests 1 $ H.property $ do
        let parameters = Parameters (NN.Dense 2 3) 4 :: Parameters T.Tensor Double
            gradients = Parameters (NN.Dense 1 2) 3 :: Parameters T.Tensor Double
            config = NN.SGD 0.1
            (updated, state) = NN.optimizerStep config parameters gradients (NN.initialState config parameters)
        state H.=== ()
        approxList (values (NN.denseWeights (projection updated))) [1.9]
        approxList (values (NN.denseBias (projection updated))) [2.8]
        approxList (values (scale updated)) [3.7],
      testProperty "SGD updates scalar numbers and preserves Float tensors and empty tensors" $ H.withTests 1 $ H.property $ do
        let emptyState = ()
            (scalar, _) = NN.optimizerStep (NN.SGD 0.1) (3 :: Double) 6 emptyState
            (floatTensor, _) = NN.optimizerStep (NN.SGD 0.5) (T.full @'[2] (3 :: Float)) 2 emptyState
            (empty, _) = NN.optimizerStep (NN.SGD 0.1) (T.zeros @'[0] @Double) 1 emptyState
        approxList [scalar] [2.4]
        T.toList (T.run T.CPU floatTensor) H.=== [2, 2]
        values empty H.=== [],
      testProperty "SGD applies gradients from a dense regression objective" $ H.withTests 1 $ H.property $ do
        let parameters = Parameters (NN.Dense 0 0) 1 :: Parameters T.Tensor Double
            input = T.fromList @'[1] [2 :: Double]
            target = T.full @'[1] (4 :: Double)
            objective :: (AG.Context s) => Parameters (AG.Tensor s) Double -> AG.Tensor s '[] Double
            objective p = NN.mse T.sum (NN.dense (projection p) (AG.constant input) * T.broadcast (scale p)) (AG.constant target)
            (loss, gradients) = AG.backprop objective parameters
            config = NN.SGD 0.1
            (updated, _) = NN.optimizerStep config parameters gradients (NN.initialState config parameters)
        values loss H.=== [16]
        values (NN.denseWeights (projection gradients)) H.=== [-16]
        values (NN.denseBias (projection gradients)) H.=== [-8]
        values (scale gradients) H.=== [0]
        approxList (values (AG.evaluate objective updated)) [0],
      testProperty "SGD supports manual records without Generic" $ H.withTests 1 $ H.property $ do
        let parameters = Manual (T.fromList @'[2] [1, 2]) 3
            objective p = (T.sum (weightVariables p * weightVariables p), biasVariable p * biasVariable p)
            (_, gradients) = AG.backprop objective parameters
            config = NN.SGD 0.1
            (updated, ()) = NN.optimizerStep config parameters gradients (NN.initialState config parameters)
        values (weights gradients) H.=== [2, 4]
        bias gradients H.=== 6
        approxList (values (weights updated)) [0.8, 1.6]
        approxList [bias updated] [2.4],
      testProperty "manual field registration derives constraints from ordinary Generic records" $ H.withTests 1 $ H.property $ do
        let parameters = GenericManual (T.fromList @'[2] [1, 2]) 3
            objective p = (T.sum (genericWeightVariables p * genericWeightVariables p), genericBiasVariable p * genericBiasVariable p)
            (_, gradients) = AG.backprop objective parameters
            config = NN.SGD 0.1
            (updated, _) = NN.optimizerStep config parameters gradients (NN.initialState config parameters)
        approxList (values (genericWeights updated)) [0.8, 1.6]
        approxList [genericBias updated] [2.4],
      testProperty "zero learning rates preserve parameters and can be changed between steps" $ H.withTests 1 $ H.property $ do
        let (unchanged, ()) = NN.optimizerStep (NN.SGD 0) (3 :: Double) 2 ()
            (updated, ()) = NN.optimizerStep (NN.SGD 0.2) unchanged 2 ()
        unchanged H.=== 3
        approxList [updated] [2.6],
      testProperty "SGD repeatedly trains a prepared scalar tensor objective" $ H.withTests 1 $ H.property $ do
        let execute = T.run1 T.CPU $ \p ->
              let gradient = AG.gradient (\x -> (x - 3) * (x - 3)) p
               in fst (NN.optimizerStep (NN.SGD 0.25) p gradient ())
            initial = T.run T.CPU (T.full @'[] (0 :: Double))
            final = iterate execute initial !! 12
        approxList (T.toList final) [3 * (1 - 0.5 ** 12)],
      testProperty "SGD rejects integral leaves" $
        H.withTests 1 $
          Compiler.rejects "test/NNTest/Fixtures/Rejected/SGDInt.hs" "Fractional Int",
      testProperty "manual registration cannot omit a field constraint" $
        H.withTests 1 $
          Compiler.rejects "test/AutogradTest/Differentiable/Fixtures/Rejected/MissingConstraint.hs" "c Float"
    ]
