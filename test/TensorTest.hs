module TensorTest (tests) where

import qualified TensorTest.Backend
import qualified TensorTest.Construction
import qualified TensorTest.Dimensions
import qualified TensorTest.Elementwise
import qualified TensorTest.Math
import qualified TensorTest.Shape
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests =
  testGroup
    "tensor"
    [ TensorTest.Dimensions.tests,
      TensorTest.Construction.tests,
      TensorTest.Elementwise.tests,
      TensorTest.Math.tests,
      TensorTest.Shape.tests,
      TensorTest.Backend.tests
    ]
