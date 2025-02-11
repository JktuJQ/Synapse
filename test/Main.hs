-- | Test suite for @Synapse@ library.


module Main (main) where


import qualified LinearAlgebraTest
import qualified AutogradTest

import Test.HUnit (Test(TestList), runTestTTAndExit)


main :: IO ()
main = runTestTTAndExit $ TestList
    [ LinearAlgebraTest.tests
    , AutogradTest.tests
    ]
