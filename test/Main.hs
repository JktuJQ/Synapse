module Main (main) where

import qualified TensorTest
import Test.Tasty (defaultMain, testGroup)

main :: IO ()
main =
  defaultMain $
    testGroup
      "synapse"
      [ TensorTest.tests
      ]
