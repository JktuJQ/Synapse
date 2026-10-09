module TensorTest.Shapes (tests) where

import qualified TensorTest.Shapes.Dimensions
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests =
  testGroup
    "shapes"
    [ TensorTest.Shapes.Dimensions.tests
    ]
