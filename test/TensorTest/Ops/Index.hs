module TensorTest.Ops.Index (tests) where

import qualified TensorTest.Ops.Index.Operations
import qualified TensorTest.Ops.Index.Types
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests = testGroup "index" [TensorTest.Ops.Index.Operations.tests, TensorTest.Ops.Index.Types.tests]
