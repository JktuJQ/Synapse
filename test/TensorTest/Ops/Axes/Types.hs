{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Axes.Types (tests) where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (natVal)
import qualified Hedgehog as H
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Axes as Axes
import qualified TensorTest.Compiler as Compiler
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup "types" $
    testProperty "type-level axis transformations" (H.withTests 1 propAxisTypes)
      : testProperty "axis selection retains its order" (H.withTests 1 propAxisSelection)
      : testProperty "valid axes compile" (H.withTests 1 validAxes)
      : [ testProperty name (H.withTests 1 (rejects file diagnostic))
          | (name, file, diagnostic) <-
              [ ("axis equal to rank", "AxisBounds", "Axis 3 is out of bounds"),
                ("scalar has no axes", "ScalarAxis", "Axis 0 is out of bounds"),
                ("cannot remove missing axis", "RemoveBounds", "Cannot remove axis 2"),
                ("cannot insert beyond rank", "InsertBounds", "Cannot insert axis 3"),
                ("selection rejects duplicates", "SelectDuplicate", "Repeated axis 1"),
                ("selection rejects missing axes", "SelectBounds", "Axis 3 is out of bounds"),
                ("selection constraint rejects duplicates", "SelectConstraintDuplicate", "Repeated axis 1"),
                ("selection constraint rejects missing axes", "SelectConstraintBounds", "Axis 3 is out of bounds"),
                ("permutation rejects duplicates", "PermuteDuplicate", "Repeated axis 0"),
                ("permutation rejects missing axes", "PermuteBounds", "Axis 2 is out of bounds"),
                ("permutation rejects too few axes", "PermuteShort", "must contain every axis"),
                ("permutation rejects too many axes", "PermuteLong", "must contain every axis"),
                ("squeeze rejects non-singleton axes", "SqueezeSize", "expected size 1, got 2"),
                ("squeeze rejects zero-size axes", "SqueezeZero", "expected size 1, got 0"),
                ("squeeze rejects missing axes", "SqueezeBounds", "Cannot remove axis 2"),
                ("unsqueeze rejects missing insertion positions", "UnsqueezeBounds", "Cannot insert axis 3")
              ]
        ]

validAxes :: H.Property
validAxes = Compiler.accepts "test/TensorTest/Ops/Axes/Fixtures/Accepted/Valid.hs"

rejects :: String -> String -> H.Property
rejects file = Compiler.rejects ("test/TensorTest/Ops/Axes/Fixtures/Rejected/" ++ file ++ ".hs")

propAxisTypes :: H.Property
propAxisTypes = H.property $ do
  natVal (Proxy @(Axes.Rank '[2, 3, 4])) H.=== 3
  natVal (Proxy @(Axes.Rank '[])) H.=== 0
  natVal (Proxy @(Axes.AxisSize 0 '[2, 3, 4])) H.=== 2
  natVal (Proxy @(Axes.AxisSize 2 '[2, 3, 4])) H.=== 4
  T.shape @(Axes.RemoveAxis 0 '[2, 3, 4]) H.=== [3, 4]
  T.shape @(Axes.RemoveAxis 1 '[2, 3, 4]) H.=== [2, 4]
  T.shape @(Axes.RemoveAxis 2 '[2, 3, 4]) H.=== [2, 3]
  T.shape @(Axes.RemoveAxis 0 '[1]) H.=== []
  natVal (Proxy @(Axes.AxisSize 1 '[2, 0, 3])) H.=== 0
  T.shape @(Axes.InsertAxis 0 5 '[2, 3]) H.=== [5, 2, 3]
  T.shape @(Axes.InsertAxis 1 5 '[2, 3]) H.=== [2, 5, 3]
  T.shape @(Axes.InsertAxis 2 5 '[2, 3]) H.=== [2, 3, 5]
  T.shape @(Axes.InsertAxis 0 1 '[]) H.=== [1]
  T.shape @(Axes.InsertAxis 1 0 '[2, 3]) H.=== [2, 0, 3]

propAxisSelection :: H.Property
propAxisSelection = H.property $ do
  Axes.axes @'[2, 0] H.=== [2, 0]
  Axes.axes @'[] H.=== []
  T.shape @(Axes.SelectAxes '[2, 0] '[2, 3, 4]) H.=== [4, 2]
  T.shape @(Axes.SelectAxes '[] '[2, 3, 4]) H.=== []
  T.shape @(Axes.PermuteAxes '[2, 0, 1] '[2, 3, 4]) H.=== [4, 2, 3]
  T.shape @(Axes.PermuteAxes '[] '[]) H.=== []
