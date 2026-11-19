module TensorTest.Ops.Reduction (tests) where

import qualified TensorTest.Ops.Reduction.Masked
import qualified TensorTest.Ops.Reduction.Operations
import qualified TensorTest.Ops.Reduction.Types
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests =
    testGroup
        "reduction"
        [ TensorTest.Ops.Reduction.Operations.tests
        , TensorTest.Ops.Reduction.Types.tests
        , TensorTest.Ops.Reduction.Masked.tests
        ]
