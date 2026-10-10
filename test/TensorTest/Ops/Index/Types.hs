{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Ops.Index.Types (tests) where

import qualified Hedgehog as H
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Index as I
import qualified TensorTest.Compiler as Compiler
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup "types" $
    [ testProperty "slice shapes reflect removed and retained axes" (H.withTests 1 propShapes),
      testProperty "valid indexing compiles" (H.withTests 1 (Compiler.accepts (root ++ "Accepted/Valid.hs")))
    ]
      ++ [ testProperty file (H.withTests 1 (Compiler.rejects (root ++ "Rejected/" ++ file ++ ".hs") diagnostic))
           | (file, diagnostic) <-
               [ ("IndexBounds", "Index 3 is out of bounds for axis 1"),
                 ("IndexShort", "Index has too few coordinates"),
                 ("IndexLong", "Index has too many coordinates"),
                 ("IndexEmpty", "Index 0 is out of bounds for axis 0"),
                 ("SliceShort", "Slice must describe every tensor axis"),
                 ("SliceLong", "Slice must describe every tensor axis"),
                 ("SliceAtBounds", "Index 3 is out of bounds for axis 1"),
                 ("SliceStartBounds", "Slice boundary 5 is out of bounds"),
                 ("SliceStopBounds", "Slice boundary 5 is out of bounds"),
                 ("SliceZeroStep", "Slice step must be positive"),
                 ("ReverseStartBounds", "Slice boundary 5 is out of bounds"),
                 ("ReverseStopBounds", "Slice boundary 5 is out of bounds"),
                 ("ReverseZeroStep", "Slice step must be positive")
               ]
         ]
  where
    root = "test/TensorTest/Ops/Index/Fixtures/"

propShapes :: H.Property
propShapes = H.property $ do
  T.shape @(I.SliceShape '[ 'T.All, 'T.At 1, 'T.Range 0 7 2] '[2, 3, 7]) H.=== [2, 4]
  T.shape @(I.SliceShape '[ 'T.Range 1 2 1, 'T.All] '[3, 4]) H.=== [1, 4]
  T.shape @(I.SliceShape '[ 'T.At 1, 'T.At 2] '[2, 3]) H.=== []
  T.shape @(I.SliceShape '[ 'T.RangeReverse 0 4 1] '[4]) H.=== [4]
  T.shape @(I.SliceShape '[ 'T.Range 3 1 1] '[4]) H.=== [0]
  T.shape @(I.SliceShape '[ 'T.RangeReverse 1 7 2] '[8]) H.=== [3]
  T.shape @(I.SliceShape '[ 'T.RangeReverse 3 1 2] '[4]) H.=== [0]
