{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Dimensions (tests) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (natVal)
import qualified Hedgehog as H
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Dimensions as D
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "dimensions"
    [ testProperty "scalar and empty dimensions" (H.withTests 1 propDimensionEdges),
      testProperty "backend dimensions preserve axis order" (H.withTests 1 propDimensionOrder),
      testProperty "type-level size and reverse" (H.withTests 1 propTypeDimensions),
      testProperty "shape exposes type-level dimensions" (H.withTests 1 propShape),
      testProperty "shape supports rank above nine" (H.withTests 1 propShapeAboveNine),
      testProperty "shapeSize multiplies dimensions" (H.withTests 1 propShapeSize)
    ]

propShape :: H.Property
propShape = H.property $ do
  T.shape @'[2, 3, 4] H.=== [2, 3, 4]

propShapeAboveNine :: H.Property
propShapeAboveNine = H.property $ do
  T.shape @'[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] H.=== [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

propShapeSize :: H.Property
propShapeSize = H.property $ do
  T.shapeSize @'[2, 3, 4] H.=== 24

propDimensionEdges :: H.Property
propDimensionEdges = H.property $ do
  T.shape @'[] H.=== []
  T.shapeSize @'[] H.=== 1
  T.shape @'[2, 0, 3] H.=== [2, 0, 3]
  T.shapeSize @'[2, 0, 3] H.=== 0

propDimensionOrder :: H.Property
propDimensionOrder = H.property $ do
  D.shapeList (Proxy @'[2, 3, 4]) H.=== [2, 3, 4]
  D.shapeVal (Proxy @'[2, 3, 4]) H.=== (A.Z A.:. (2 :: Int) A.:. (3 :: Int) A.:. (4 :: Int))

propTypeDimensions :: H.Property
propTypeDimensions = H.property $ do
  natVal (Proxy @(D.ShapeSize '[2, 3, 4])) H.=== 24
  natVal (Proxy @(D.ShapeSize '[])) H.=== 1
  natVal (Proxy @(D.ShapeSize '[2, 0, 3])) H.=== 0
  D.shape @(D.Reverse '[2, 3, 4]) H.=== [4, 3, 2]
