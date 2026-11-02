module TensorTest.Ops (tests) where

import qualified TensorTest.Ops.Convolution
import qualified TensorTest.Ops.Elementwise
import qualified TensorTest.Ops.Index
import qualified TensorTest.Ops.Math
import qualified TensorTest.Ops.Reduction
import qualified TensorTest.Ops.Shape
import qualified TensorTest.Ops.Spatial
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests =
  testGroup
    "operations"
    [ TensorTest.Ops.Spatial.tests,
      TensorTest.Ops.Convolution.tests,
      TensorTest.Ops.Elementwise.tests,
      TensorTest.Ops.Math.tests,
      TensorTest.Ops.Index.tests,
      TensorTest.Ops.Shape.tests,
      TensorTest.Ops.Reduction.tests
    ]
