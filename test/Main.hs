-- | Test suite for @Synapse@ library.


module Main (main) where


import qualified LinearAlgebraTest

import Test.HUnit (runTestTTAndExit)


main :: IO ()
main = runTestTTAndExit LinearAlgebraTest.tests
