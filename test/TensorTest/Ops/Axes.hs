module TensorTest.Ops.Axes (tests) where

import qualified TensorTest.Ops.Axes.Operations
import qualified TensorTest.Ops.Axes.Types
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests =
  testGroup
    "axes"
    [ TensorTest.Ops.Axes.Operations.tests,
      TensorTest.Ops.Axes.Types.tests
    ]
