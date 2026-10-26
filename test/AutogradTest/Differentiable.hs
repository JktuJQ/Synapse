{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module AutogradTest.Differentiable (tests) where

import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

newtype Single (t :: T.TensorLike) = Single {singleValue :: t '[] Double} deriving (Generic)

type instance T.WithTensor t (Single T.Tensor) = Single t

data Wide (t :: T.TensorLike) = Wide
  { first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelfth :: t '[] Double
  }
  deriving (Generic)

type instance T.WithTensor t (Wide T.Tensor) = Wide t

components :: Wide t -> [t '[] Double]
components p = [first p, second p, third p, fourth p, fifth p, sixth p, seventh p, eighth p, ninth p, tenth p, eleventh p, twelfth p]

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "tensor structures"
    [ testProperty "single-field records need only Generic and WithTensor" $ H.withTests 1 $ H.property $ do
        let input = Single (T.full @'[] (3 :: Double))
            function (Single x) = Single (x * x)
            (Single output, Single gradient) = AG.backprop function input
            Single evaluated = AG.evaluate function input
        values output H.=== [9]
        values gradient H.=== [6]
        values evaluated H.=== [9],
      testProperty "records support more than nine independently registered tensors" $ H.withTests 1 $ H.property $ do
        let input :: Wide T.Tensor
            input = Wide 1 2 3 4 5 6 7 8 9 10 11 12
            (result, gradients) = AG.backprop (sum . components) input
        values result H.=== [78]
        map values (components gradients) H.=== replicate 12 [1],
      testProperty "unused record fields receive zero gradients" $ H.withTests 1 $ H.property $ do
        let input :: Wide T.Tensor
            input = Wide 1 2 3 4 5 6 7 8 9 10 11 12
            gradients = AG.gradient (\p -> first p + twelfth p) input
        map values (components gradients) H.=== [[1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1]],
      testProperty "the same correspondence materializes wide records" $ H.withTests 1 $ H.property $ do
        let input :: Wide T.Tensor
            input = Wide 1 2 3 4 5 6 7 8 9 10 11 12
            output = T.run T.CPU input
        map T.toList (components output) H.=== map (: []) [1 .. 12],
      testProperty "ordinary numbers are not differentiable leaves" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "Number.hs") "Double",
      testProperty "element expressions are not differentiable leaves" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "Element.hs") "TensorElement",
      testProperty "integer tensor arithmetic requires fractional gradient elements" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "IntegerArithmetic.hs") "Fractional",
      testProperty "integer tensor reshaping requires fractional gradient elements" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "IntegerReshape.hs") "Fractional",
      testProperty "argmax has no differentiable result" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "ArgmaxGradient.hs") "Couldn't match type",
      testProperty "argmin has no differentiable result" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "ArgminGradient.hs") "Couldn't match type"
    ]
  where
    fixtures = "test/AutogradTest/Differentiable/Fixtures/Rejected/"
