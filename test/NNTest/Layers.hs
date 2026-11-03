module NNTest.Layers (tests) where

import qualified NNTest.Layers.Convolution as Convolution
import qualified NNTest.Layers.Dense as Dense
import qualified NNTest.Layers.Dropout as Dropout
import qualified NNTest.Layers.Embedding as Embedding
import qualified NNTest.Layers.Normalization as Normalization
import qualified NNTest.Layers.Recurrent as Recurrent
import Test.Tasty (TestTree, testGroup)

tests :: TestTree
tests = testGroup "layers" [Dense.tests, Convolution.tests, Normalization.tests, Dropout.tests, Embedding.tests, Recurrent.tests]
