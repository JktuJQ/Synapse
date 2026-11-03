{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module NNTest.Gradient (tests) where

import Control.Monad (forM_)
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh, T.DType a) => T.Tensor sh a -> [a]
values = T.toList . T.run T.CPU

-- Mixed precision, nested products and different tensor ranks in one record.
data Gradients t = Gradients
  { matrix :: t '[2, 2] Double,
    nested :: (t '[1] Float, t '[] Double)
  }
  deriving (Generic)

type instance T.WithTensor t (Gradients T.Tensor) = Gradients t

gradients :: Gradients T.Tensor
gradients = Gradients (T.fromList [3, -4, 0, 0]) (T.fromList [12], 0)

components :: Gradients T.TensorData -> [Double]
components (Gradients x (y, z)) = T.toList x ++ map realToFrac (T.toList y) ++ T.toList z

tests :: TestTree
tests =
  testGroup
    "gradient norms"
    [ testProperty "tensor norms reduce all entries and support integers and Float" $ H.withTests 1 $ H.property $ do
        let x = T.fromList @'[2, 2] [3, -4, 0, 12 :: Double]
        values (T.normL1 x) H.=== [19]
        approxList (values $ T.normL2 x) [13]
        values (T.normL1 $ T.fromList @'[3] [-3, 0, 4 :: Int]) H.=== [7]
        approxList (values $ T.normL2 $ T.fromList @'[2] [-3, 4 :: Float]) [5]
        values (T.normL1 $ T.full @'[] (-7 :: Double)) H.=== [7]
        approxList (values $ T.normL2 $ T.full @'[] (-7 :: Double)) [7],
      testProperty "tensor norms and pullbacks are zero for empty and all-zero inputs" $ H.withTests 1 $ H.property $ do
        let empty = T.zeros @'[0, 2] @Double
            zero = T.zeros @'[3] @Double
            (l1, dl1) = AG.backprop T.normL1 zero
            (l2, dl2) = AG.backprop T.normL2 zero
            (emptyNorm, emptyGradient) = AG.backprop T.normL2 empty
        values (T.normL1 empty) H.=== [0]
        values emptyNorm H.=== [0]
        values emptyGradient H.=== []
        forM_ [l1, l2] $ \norm -> values norm H.=== [0]
        forM_ [dl1, dl2] $ \gradient -> values gradient H.=== [0, 0, 0],
      testProperty "tensor norm pullbacks preserve scalar seeds and choose zero at zero entries" $ H.withTests 1 $ H.property $ do
        let x = T.fromList @'[3] [-3, 0, 4 :: Double]
            (_, l1Backward) = AG.pullback T.normL1 x
            (_, l2Backward) = AG.pullback T.normL2 x
        approxList (values $ l1Backward 2) [-2, 0, 2]
        approxList (values $ l2Backward 2) [-1.2, 0, 1.6],
      testProperty "L2 norms and gradients avoid intermediate overflow and underflow" $ H.withTests 1 $ H.property $ do
        forM_ [1e200, 1e-200] $ \scale -> do
          let x = T.fromList @'[2] [3 * scale, 4 * scale :: Double]
              (norm, gradient) = AG.backprop T.normL2 x
          approxList (map (/ scale) $ values norm) [5]
          approxList (values gradient) [0.6, 0.8]
        forM_ [1e30, 1e-30] $ \scale -> do
          let norm = T.normL2 $ T.fromList @'[2] [3 * scale, 4 * scale :: Float]
          approxList (map (/ scale) $ values norm) [5],
      testProperty "tensor norms propagate NaN and return infinity for infinite inputs" $ H.withTests 1 $ H.property $ do
        let infinite = T.fromList @'[2] [1 / 0, 3 :: Double]
            nan = T.fromList @'[2] [1 / 0, 0 / 0 :: Double]
        H.assert $ all isInfinite $ values $ T.normL1 infinite
        H.assert $ all isInfinite $ values $ T.normL2 infinite
        H.assert $ all isNaN $ values $ T.normL1 nan
        H.assert $ all isNaN $ values $ T.normL2 nan,
      testProperty "global norms combine all fields of nested mixed-precision records" $ H.withTests 1 $ H.property $ do
        approxList (values $ NN.gradientNormL1 gradients) [19]
        approxList (values $ NN.gradientNormL2 gradients) [13]
        approxList (values $ NN.gradientNormL2 (T.fromList [3, 4] :: T.Tensor '[2] Float)) [5]
        let emptyFields :: (T.Tensor '[0] Float, T.Tensor '[2, 0] Double)
            emptyFields = (T.zeros, T.zeros)
        values (NN.gradientNormL1 emptyFields) H.=== [0]
        values (NN.gradientNormL2 emptyFields) H.=== [0],
      testProperty "clipping uses one global factor rather than clipping each tensor separately" $ H.withTests 1 $ H.property $ do
        let l1 = NN.clipGradNormL1 9.5 gradients
            l2 = NN.clipGradNormL2 6.5 gradients
            expected = [1.5, -2, 0, 0, 6, 0]
        approxList (components $ T.run T.CPU l1) expected
        approxList (components $ T.run T.CPU l2) expected
        approxList (values $ NN.gradientNormL1 l1) [9.5]
        approxList (values $ NN.gradientNormL2 l2) [6.5],
      testProperty "small gradients are preserved and a zero limit clears nonzero gradients" $ H.withTests 1 $ H.property $ do
        let unchanged = components $ T.run T.CPU gradients
        approxList (components $ T.run T.CPU $ NN.clipGradNormL1 20 gradients) unchanged
        approxList (components $ T.run T.CPU $ NN.clipGradNormL2 20 gradients) unchanged
        approxList (components $ T.run T.CPU $ NN.clipGradNormL1 0 gradients) $ replicate 6 0
        approxList (components $ T.run T.CPU $ NN.clipGradNormL2 0 gradients) $ replicate 6 0,
      testProperty "zero and empty gradients remain finite even with a zero limit" $ H.withTests 1 $ H.property $ do
        let zero = Gradients 0 (0, 0)
            empty = (T.zeros @'[0] @Float, T.zeros @'[2, 0] @Double)
        forM_ [NN.clipGradNormL1, NN.clipGradNormL2] $ \clip ->
          approxList (components $ T.run T.CPU $ clip 0 zero) $ replicate 6 0
        let (first, second) = T.run T.CPU $ NN.clipGradNormL2 0 empty
        T.toList first H.=== []
        T.toList second H.=== [],
      testProperty "global L2 aggregation remains stable across large and tiny tensor fields" $ H.withTests 1 $ H.property $ do
        forM_ [1e200, 1e-200] $ \scale -> do
          let g = (T.full @'[] (3 * scale :: Double), T.full @'[] (4 * scale :: Double))
              clipped = NN.clipGradNormL2 (2.5 * scale) g
              (first, second) = T.run T.CPU clipped
          approxList (map (/ scale) $ values $ NN.gradientNormL2 g) [5]
          approxList (map (/ scale) $ T.toList first ++ T.toList second) [1.5, 2],
      testProperty "Float clipping does not underflow the scale before multiplication" $ H.withTests 1 $ H.property $ do
        let g = T.full @'[1] (1e30 :: Float)
            clipped = NN.clipGradNormL2 1e-20 g
        approxList (map (/ 1e-20) $ values clipped) [1],
      testProperty "prepared differentiation clipping and SGD accept changing parameters" $ H.withTests 1 $ H.property $ do
        let step = T.run1 T.CPU $ \p ->
              let gradient = AG.gradient (\(x, y) -> (x * x + y * y) / 2) p
                  clipped = NN.clipGradNormL2 1 gradient
               in (NN.gradientNormL2 gradient, NN.evalState (NN.optimizerStep (NN.SGD 1) p clipped) ())
            first = (T.fromListData @'[] [3 :: Double], T.fromListData @'[] [4 :: Double])
            second = (T.fromListData @'[] [6 :: Double], T.fromListData @'[] [8 :: Double])
            (firstNorm, (firstX, firstY)) = step first
            (secondNorm, (secondX, secondY)) = step second
        approxList (T.toList firstNorm) [5]
        approxList (T.toList firstX ++ T.toList firstY) [2.4, 3.2]
        approxList (T.toList secondNorm) [10]
        approxList (T.toList secondX ++ T.toList secondY) [5.4, 7.2],
      testProperty "gradient clipping rejects integral tensor fields" $
        H.withTests 1 $
          Compiler.rejects "test/NNTest/Fixtures/Rejected/IntegralGradient.hs" "GradientTensor"
    ]
