module NNTest.Layers (tests) where

import qualified NNTest.Layers.Convolution as Convolution
import qualified NNTest.Layers.Dense as Dense
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests = testGroup "layers" [Dense.tests, Convolution.tests]
