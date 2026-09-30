module Main (main) where

import Test.Tasty (defaultMain, testGroup)

import qualified TensorTest

main :: IO ()
main =
  defaultMain $
    testGroup
      "synapse"
      [ TensorTest.tests
      ]
