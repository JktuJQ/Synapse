module Main (main) where

import qualified AutogradTest
import qualified TensorTest
import Test.Tasty (defaultMain, testGroup)

main :: IO ()
main =
  defaultMain $
    testGroup
      "synapse"
      [ TensorTest.tests,
        AutogradTest.tests
      ]
