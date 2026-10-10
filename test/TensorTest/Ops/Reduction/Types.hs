{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Reduction.Types (tests) where

import qualified Hedgehog as H
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Axes as Axes
import qualified TensorTest.Compiler as Compiler
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests = testGroup "types" $
  [ testProperty "reduced shapes preserve axis order" (H.withTests 1 propShapes),
    testProperty "valid reductions compile" (H.withTests 1 (Compiler.accepts (root ++ "Accepted/Valid.hs")))
  ] ++
  [ testProperty file (H.withTests 1 (Compiler.rejects (root ++ "Rejected/" ++ file ++ ".hs") diagnostic))
    | (file, diagnostic) <-
      [ ("DuplicateAxes", "Repeated axis 1"),
        ("MissingAxis", "Axis 2 is out of bounds"),
        ("EmptyMinimum", "Cannot reduce empty tensor"),
        ("EmptyMaximum", "Cannot reduce empty tensor"),
        ("EmptyMinimumAxis", "Cannot reduce empty axis 1"),
        ("EmptyMaximumAxis", "Cannot reduce empty axis 0"),
        ("IntegerMean", "RealFloat Int")
      ]
  ]
  where
    root = "test/TensorTest/Ops/Reduction/Fixtures/"

propShapes :: H.Property
propShapes = H.property $ do
  T.shape @(Axes.ReduceShape '[0, 2] '[2, 3, 4]) H.=== [3]
  T.shape @(Axes.ReduceShape '[2, 0] '[2, 3, 4]) H.=== [3]
  T.shape @(Axes.ReduceShape '[1] '[2, 3, 4]) H.=== [2, 4]
  T.shape @(Axes.ReduceShape '[2, 0, 1] '[2, 3, 4]) H.=== []
  T.shape @(Axes.ReduceShape '[] '[2, 3, 4]) H.=== [2, 3, 4]
  T.shape @(Axes.ReduceShape '[] '[]) H.=== []
