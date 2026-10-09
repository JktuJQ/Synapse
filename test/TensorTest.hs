module TensorTest (tests) where

import qualified TensorTest.Backend
import qualified TensorTest.Construction
import qualified TensorTest.Conversion
import qualified TensorTest.Ops
import qualified TensorTest.Shapes
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests =
  testGroup
    "tensor"
    [ TensorTest.Backend.tests,
      TensorTest.Construction.tests,
      TensorTest.Conversion.tests,
      TensorTest.Ops.tests,
      TensorTest.Shapes.tests
    ]
