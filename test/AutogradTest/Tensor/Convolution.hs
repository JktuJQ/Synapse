{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module AutogradTest.Tensor.Convolution (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import Synapse.Tensor.Shapes.Convolution (Conv2dShape)
import TensorTest.Ops.Convolution.Reference (reference)
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

check :: forall stride padding input kernel. (T.Shape stride, T.Shape padding, T.Shape input, T.Shape kernel, T.Shape (Conv2dShape input kernel stride padding '[1, 1] 1)) => H.Property
check = H.withTests 1 $ H.property $ do
  let xs :: [Double]
      xs = take (T.shapeSize @input) $ cycle [0.2, -0.3, 0.7, 0.1, -0.5]
      ws = take (T.shapeSize @kernel) $ cycle [-0.1, 0.6, 0.4, -0.2]
      seeds = take (T.shapeSize @(Conv2dShape input kernel stride padding '[1, 1] 1)) $ cycle [0.3, -0.5, 0.8, 0.2]
      input = T.fromList @input xs
      weights = T.fromList @kernel ws
      (output, backward) = AG.pullback (uncurry $ T.conv2d @stride @padding @'[1, 1] @1) (input, weights)
      (dx, dw) = backward $ T.fromList seeds
      objective x w = sum $ zipWith (*) seeds $ reference (T.shape @input) (T.shape @kernel) (T.shape @stride) (T.shape @padding) x w
      differences f values =
        let perturb i delta = zipWith (\j value -> if i == j then value + delta else value) [0 ..] values
         in [(f (perturb i 1e-5) - f (perturb i (-1e-5))) / 2e-5 | i <- [0 .. length values - 1]]
      (result, inputGradient, weightGradient) = T.run T.CPU (output, dx, dw)
  approxList (T.toList result) $ reference (T.shape @input) (T.shape @kernel) (T.shape @stride) (T.shape @padding) xs ws
  approxList (T.toList inputGradient) $ differences (`objective` ws) xs
  approxList (T.toList weightGradient) $ differences (objective xs) ws

propPrepared :: H.Property
propPrepared = H.withTests 1 $ H.property $ do
  let execute = T.run2 T.CPU $ \input weights ->
        let (output, (inputGradient, weightGradient)) = AG.backprop (uncurry $ T.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] @1) (input, weights)
         in (output, inputGradient, weightGradient)
      (first, dx, dw) = execute (T.fromListData @'[1, 1, 2, 2] [1 .. 4 :: Double]) (T.fromListData @'[1, 1, 1, 1] [2])
      (second, dx2, dw2) = execute (T.fromListData [2 .. 5]) (T.fromListData [3])
  T.toList first H.=== [2, 4, 6, 8]
  T.toList dx H.=== replicate 4 2
  T.toList dw H.=== [10]
  T.toList second H.=== [6, 9, 12, 15]
  T.toList dx2 H.=== replicate 4 3
  T.toList dw2 H.=== [14]

propShared :: H.Property
propShared = H.withTests 1 $ H.property $ do
  let input = T.fromList @'[1, 1, 1, 1] [3 :: Double]
      (_, gradient) = AG.backprop (\x -> T.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] @1 x x) input
  T.toList (T.run T.CPU gradient) H.=== [6]

tests :: TestTree
tests =
  testGroup
    "convolution gradients"
    [ testProperty "overlapping rectangular windows and multiple batches" $ check @'[1, 1] @'[0, 0, 0, 0] @'[2, 2, 3, 4] @'[2, 2, 2, 3],
      testProperty "strides and asymmetric padding respect nonuniform seeds" $ check @'[2, 3] @'[2, 0, 1, 2] @'[1, 2, 3, 4] @'[2, 2, 3, 2],
      testProperty "large padding drops contributions outside the input" $ check @'[2, 1] @'[3, 2, 4, 1] @'[1, 1, 1, 2] @'[2, 1, 2, 3],
      testProperty "empty batches return zero weight gradients" $ check @'[1, 1] @'[0, 0, 0, 0] @'[0, 2, 3, 3] @'[2, 2, 2, 2],
      testProperty "empty output channels return zero input gradients" $ check @'[1, 1] @'[0, 0, 0, 0] @'[1, 2, 3, 3] @'[0, 2, 2, 2],
      testProperty "empty input channels produce zero outputs" $ check @'[1, 1] @'[0, 0, 0, 0] @'[1, 0, 3, 3] @'[2, 0, 2, 2],
      testProperty "empty padded spatial input returns zero weight gradients" $ check @'[1, 1] @'[1, 1, 1, 1] @'[1, 1, 0, 0] @'[2, 1, 1, 1],
      testProperty "prepared gradients accept changing inputs and weights" propPrepared,
      testProperty "shared inputs accumulate both gradient paths" propShared
    ]
