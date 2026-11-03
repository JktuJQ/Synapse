{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module AutogradTest.Tensor.ElementLifting (tests) where

import qualified Data.Array.Accelerate as A
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import Synapse.Tensor.Ops (Needs (..), TensorOps (..))
import qualified Synapse.Tensor.Type as Raw
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

-- Differently shaped inputs exercise independent registration and reconstruction.
data Inputs t = Inputs (t '[2] Double) (t '[3] Double) deriving (Generic)

type instance T.WithTensor t (Inputs T.Tensor) = Inputs t

-- Neither the element residuals nor host metadata are differentiable inputs.
data Cache = Cache (T.TensorElement Double) (T.TensorElement Double) Int

productElement :: (TensorOps t Double) => Inputs t -> T.Element t Double
productElement =
  customElementVJP
    Numeric
    (\(Inputs x y) -> T.index @'[0] x * T.index @'[1] y)
    (\(Inputs x y) -> let a = T.index @'[0] x; b = T.index @'[1] y in (a * b, Cache a b 1))
    ( \(Cache a b multiplier) seed ->
        let scatter index gradient = T.fromGen $ \ix -> T.select (T.coordinate @0 ix T.== T.element index) gradient 0
            scaled = fromIntegral multiplier * seed
         in Inputs (scatter 0 $ scaled * b) (scatter 1 $ scaled * a)
    )

sharedElement :: (TensorOps t Double) => (t '[2] Double, t '[2] Double) -> T.Element t Double
sharedElement =
  customElementVJP
    Numeric
    (\(x, y) -> T.index @'[0] x * T.index @'[1] y)
    (\(x, y) -> let a = T.index @'[0] x; b = T.index @'[1] y in (a * b, (a, b)))
    ( \(a, b) seed ->
        ( T.fromGen $ \ix -> T.select (T.coordinate @0 ix T.== 0) (seed * b) 0,
          T.fromGen $ \ix -> T.select (T.coordinate @0 ix T.== 1) (seed * a) 0
        )
    )

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "element primitive lifting"
    [ testProperty "records, independent shapes and a private cache preserve element cotangents" $ H.withTests 1 $ H.property $ do
        let input = Inputs (T.fromList [2, 3]) (T.fromList [5, 7, 11]) :: Inputs T.Tensor
            (output, backward) = AG.pullback (T.scalar . productElement) input
            Inputs dx dy = backward $ T.full @'[] 3
        values (T.scalar $ productElement input) H.=== [14]
        values output H.=== [14]
        values dx H.=== [21, 0]
        values dy H.=== [0, 6, 0],
      testProperty "repeated tensor arguments accumulate both element gradient paths" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [2, 3 :: Double]
            (output, gradient) = AG.backprop (\x -> T.scalar $ sharedElement (x, x)) input
        values output H.=== [6]
        values gradient H.=== [3, 2],
      testProperty "prepared element residuals follow changing runtime inputs" $ H.withTests 1 $ H.property $ do
        let execute = T.run1 T.CPU $ AG.backprop (T.scalar . productElement)
            (first, Inputs dx1 dy1) = execute $ Inputs (T.fromListData [2, 3]) (T.fromListData [5, 7, 11])
            (second, Inputs dx2 dy2) = execute $ Inputs (T.fromListData [4, 6]) (T.fromListData [10, 14, 22])
        T.toList first H.=== [14]
        T.toList second H.=== [56]
        T.toList dx1 H.=== [7, 0]
        T.toList dy1 H.=== [0, 2, 0]
        T.toList dx2 H.=== [14, 0]
        T.toList dy2 H.=== [0, 4, 0],
      testProperty "ordinary Int elements skip the cache and fractional pullback" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [2, 3 :: Int]
            output = customElementVJP Fractional (T.index @'[1]) (error "cache evaluated") (error "pullback evaluated") input
        T.toList (T.run T.CPU $ T.scalar output) H.=== [3],
      testProperty "non-numeric elements need no numeric dictionaries" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [False, True]
            output = customElementVJP Numeric (T.index @'[1]) (error "cache evaluated") (error "pullback evaluated") input
        T.toList (T.run T.CPU $ T.scalar output) H.=== [True],
      testProperty "element primitives can change the result element type" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [2, 3 :: Int]
            output = customElementVJP Numeric (\x -> T.index @'[1] x T.> 2) (error "cache evaluated") (error "pullback evaluated") input
        T.toList (T.run T.CPU $ T.scalar output) H.=== [True],
      testProperty "element expressions keep generator-local coordinate scope" $ H.withTests 1 $ H.property $ do
        let source = T.fromList @'[3] [2, 5, 11 :: Int]
            at selected =
              customElementVJP
                Numeric
                (`T.unsafeDynamicIndex` selected)
                (error "cache evaluated")
                (error "pullback evaluated")
                source
            output = T.fromGen @'[3] at
        T.toList (T.run T.CPU output) H.=== [2, 5, 11],
      testProperty "runtime indices remain inputs to prepared element gradients" $ H.withTests 1 $ H.property $ do
        let execute = T.run2 T.CPU $ \(indices :: T.Tensor '[1] Int) (tensor :: T.Tensor '[3] Double) ->
              let Raw.TensorElement i = T.index @'[0] indices
                  selected = Raw.TensorIndex $ A.index1 i
               in AG.backprop (\x -> T.scalar $ T.dynamicIndexOr 13 x selected) tensor
            input = T.fromListData [2, 5, 11]
            (valid, validGradient) = execute (T.fromListData [1]) input
            (invalid, invalidGradient) = execute (T.fromListData [3]) input
        T.toList valid H.=== [5]
        T.toList validGradient H.=== [0, 1, 0]
        T.toList invalid H.=== [13]
        T.toList invalidGradient H.=== [0, 0, 0]
    ]
