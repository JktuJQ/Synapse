{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}

module AutogradTest (tests) where

import qualified AutogradTest.Differentiable
import qualified AutogradTest.Execution
import qualified AutogradTest.Tensor
import Control.Monad (forM_)
import qualified Hedgehog as H
import qualified Numeric.Backprop.Explicit as BP
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

differentiate :: (forall s. (AG.Context s) => AG.Var s Double -> AG.Var s Double) -> Double -> (Double, Double)
differentiate = AG.backprop

close :: Double -> Double -> H.PropertyT IO ()
close actual expected = do
  H.annotateShow (actual, expected)
  H.assert (abs (actual - expected) <= 1e-5 * max 1 (abs expected))

tests :: TestTree
tests =
  testGroup
    "autograd"
    [ AutogradTest.Tensor.tests,
      AutogradTest.Execution.tests,
      AutogradTest.Differentiable.tests,
      testProperty "shared variables accumulate gradient contributions" $ H.withTests 1 $ H.property $ do
        let (value, gradient) = differentiate (\x -> x * x + x) 3
        value H.=== 12
        gradient H.=== 7,
      testProperty "constant results have zero input gradient" $ H.withTests 1 $ H.property $ do
        differentiate (const 7) 3 H.=== (7, 0),
      testProperty "arithmetic follows the chain rule" $ H.withTests 1 $ H.property $ do
        let (value, gradient) = differentiate (\x -> (x * x - 1) / (x + 2)) 3
        close value (8 / 5)
        close gradient (22 / 25),
      testProperty "floating derivatives agree with finite differences" $ H.withTests 1 $ H.property $ do
        let check :: (Double -> Double) -> (forall s. (AG.Context s) => AG.Var s Double -> AG.Var s Double) -> Double -> H.PropertyT IO ()
            check plain staged x = do
              let (value, gradient) = differentiate staged x
                  step = 1e-6
              close value (plain x)
              close gradient ((plain (x + step) - plain (x - step)) / (2 * step))
        forM_ [0.3, 0.7] $ \x -> do
          check exp exp x
          check log log x
          check sqrt sqrt x
          check sin sin x
          check cos cos x
          check tan tan x
          check asin asin x
          check acos acos x
          check atan atan x
          check sinh sinh x
          check cosh cosh x
          check tanh tanh x
          check asinh asinh x
          check acosh acosh (x + 2)
          check atanh atanh x
          check (\y -> y ** (y + 1)) (\y -> y ** (y + 1)) x
          check (\y -> logBase (y + 2) (y + 1)) (\y -> logBase (y + 2) (y + 1)) x,
      testProperty "tensor arithmetic preserves values and gradients" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3] :: T.Tensor '[3] Double
            (value, gradient) =
              BP.backprop
                (BP.ZF AG.zero)
                (BP.OF AG.one)
                (\x -> let v = AG.Tensor (AG.Var x) in AG.unVar (AG.unTensor (v * v + v)))
                input
        T.toList (T.run T.CPU value) H.=== [2, 6, 12]
        T.toList (T.run T.CPU gradient) H.=== [3, 5, 7],
      testProperty "element arithmetic preserves differentiation history" $ H.withTests 1 $ H.property $ do
        let input = T.constant (3 :: Double)
            (value, gradient) =
              BP.backprop
                (BP.ZF AG.zero)
                (BP.OF AG.one)
                (\x -> let v = AG.TensorElement (AG.Var x) in AG.unVar (AG.unTensorElement (v * v)))
                input
        T.toList (T.run T.CPU (T.scalar value)) H.=== [9]
        T.toList (T.run T.CPU (T.scalar gradient)) H.=== [6],
      testProperty "absolute value and signum use their local derivatives" $ H.withTests 1 $ H.property $ do
        forM_ [-2, 0, 2] $ \x -> do
          snd (differentiate abs x) H.=== signum x
          snd (differentiate signum x) H.=== 0
    ]
