module TensorTest.Ops (tests) where

import qualified TensorTest.Ops.Axes
import qualified TensorTest.Ops.Elementwise
import qualified TensorTest.Ops.Math
import qualified TensorTest.Ops.Shape
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests =
  testGroup
    "operations"
    [ TensorTest.Ops.Axes.tests,
      TensorTest.Ops.Elementwise.tests,
      TensorTest.Ops.Math.tests,
      TensorTest.Ops.Shape.tests
    ]
