-- | Test suite for "Synapse" library.


module Main (main) where


import qualified TensorsTest
import qualified AutogradTest
import qualified NNTest

import Test.HUnit (Test(TestList), runTestTTAndExit)


main :: IO ()
main = runTestTTAndExit $ TestList
    [ TensorsTest.tests
    , AutogradTest.tests
    , NNTest.tests
    ]
