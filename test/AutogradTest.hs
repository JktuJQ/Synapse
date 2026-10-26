{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}

module AutogradTest (tests) where

import qualified AutogradTest.Differentiable
import qualified AutogradTest.Execution
import qualified AutogradTest.Tensor
import Control.Monad (forM_)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

differentiate ::
  (forall s. (AG.Context s) => AG.Tensor s '[] Double -> AG.Tensor s '[] Double) ->
  Double ->
  ([Double], [Double])
differentiate f x =
  let (value, gradient) = T.run T.CPU (AG.backprop f (T.full @'[] x))
   in (T.toList value, T.toList gradient)

tests :: TestTree
tests =
  testGroup
    "autograd"
    [ AutogradTest.Tensor.tests,
      AutogradTest.Execution.tests,
      AutogradTest.Differentiable.tests,
      testProperty "shared scalar tensors accumulate gradient contributions" $ H.withTests 1 $ H.property $ do
        differentiate (\x -> x * x + x) 3 H.=== ([12], [7]),
      testProperty "constant results have zero input gradients" $ H.withTests 1 $ H.property $ do
        differentiate (const 7) 3 H.=== ([7], [0]),
      testProperty "arithmetic follows the chain rule" $ H.withTests 1 $ H.property $ do
        let (value, gradient) = differentiate (\x -> (x * x - 1) / (x + 2)) 3
        approxList value [8 / 5]
        approxList gradient [22 / 25],
      testProperty "floating derivatives agree with finite differences" $ H.withTests 1 $ H.property $ do
        let check :: (Double -> Double) -> (forall s. (AG.Context s) => AG.Tensor s '[] Double -> AG.Tensor s '[] Double) -> Double -> H.PropertyT IO ()
            check plain staged x = do
              let (value, gradient) = differentiate staged x
                  step = 1e-6
              approxList value [plain x]
              approxList gradient [(plain (x + step) - plain (x - step)) / (2 * step)]
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
      testProperty "absolute value and signum use their local derivatives" $ H.withTests 1 $ H.property $ do
        forM_ [-2, 0, 2] $ \x -> do
          snd (differentiate abs x) H.=== [signum x]
          snd (differentiate signum x) H.=== [0]
    ]
