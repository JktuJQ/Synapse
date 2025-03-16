-- | Tests @Synapse.Autograd@ module and its submodules.


module AutogradTest
    ( tests
    ) where


import Synapse.Autograd

import qualified Synapse.Tensors.Vec as V

import qualified Synapse.Tensors.Mat as M

import Test.HUnit


testNumOps :: Test
testNumOps = TestLabel "testIntOps" $ TestList
    [ TestCase $ assertEqual "zero gradient" 0 (unSymbol $ getGradientsOf a `wrt` b)
    , TestCase $ assertEqual "identity gradient" 1 (unSymbol $ getGradientsOf a `wrt` a)
    , TestCase $ assertEqual "composited identity gradient" 1 (unSymbol $ getGradientsOf c `wrt` c)
    , TestCase $ assertEqual "addition gradient" 1 (unSymbol $ getGradientsOf (a + b) `wrt` a)
    , TestCase $ assertEqual "subtraction gradient" (-1) (unSymbol $ getGradientsOf (a - b) `wrt` b)
    , TestCase $ assertEqual "multiplication gradient" 3 (unSymbol $ getGradientsOf (a * b) `wrt` a)
    , TestCase $ assertEqual "composed operations gradient" (4 * 4 * 3) (unSymbol $ getGradientsOf (a * (b + b) * a) `wrt` a)
    , TestCase $ assertEqual "renamed symbol gradient" (2 * 25) (unSymbol $ getGradientsOf (c * c) `wrt` c)
    ]
  where
    a = symbol "a" 4 :: Symbol Int
    b = symbol "b" 3 :: Symbol Int
    c = renameSymbol "c" ((a * a) + (b * b)) :: Symbol Int

testNthOrderGradients :: Test
testNthOrderGradients = TestLabel "testNthOrderGradients" $ TestList
    [ TestCase $ assertEqual "identity 2nd gradient" 0 (unSymbol $ nthGradient 2 a a)
    , TestCase $ assertEqual "composed operations 2nd gradient" (4 * 3) (unSymbol $ nthGradient 2 (a * (b + b) * a) a)
    , TestCase $ assertEqual "4th gradient" (unSymbol $ sin c) (unSymbol $ nthGradient 4 (sin c) c)
    ]
  where
    a = symbol "a" 4 :: Symbol Int
    b = symbol "b" 3 :: Symbol Int
    c = symbol "c" 0.5 :: Symbol Float

testVecOps :: Test
testVecOps = TestLabel "testVecOps" $ TestList
    [ TestCase $ assertEqual "vector addition gradient" (V.replicate 3 1.0) (unSymbol $ getGradientsOf (a + b) `wrt` a)
    , TestCase $ assertEqual "vector elementwise multiplication gradient" (unSymbol b) (unSymbol $ getGradientsOf (a * b) `wrt` a)
    , TestCase $ assertEqual "vector op gradient" (V.map cos $ unSymbol a) (unSymbol $ getGradientsOf (sin a) `wrt` a)
    ]
  where
    a = symbol "a" (V.fromList [1.0, 2.0, 3.0]) :: SymbolVec Float
    b = symbol "b" (V.fromList [3.0, 2.0, 1.0]) :: SymbolVec Float

testMatOps :: Test
testMatOps = TestLabel "testMatOps" $ TestList
    [ TestCase $ assertEqual "matrix addition gradient" (M.replicate (3, 3) 1.0) (unSymbol $ getGradientsOf (a + b) `wrt` a)
    , TestCase $ assertEqual "matrix elementwise multiplication gradient" (unSymbol b) (unSymbol $ getGradientsOf (a * b) `wrt` a)
    , TestCase $ assertEqual "matrix op gradient" (M.map cos $ unSymbol a) (unSymbol $ getGradientsOf (sin a) `wrt` a)
    , TestCase $ assertEqual "matrix transposing gradient" (M.transpose $ unSymbol c) (unSymbol $ getGradientsOf (transpose c) `wrt` c)
    , TestCase $ assertEqual "matrix multiplication+addition gradient" (M.transpose $ unSymbol c) (unSymbol $ getGradientsOf (c `matMul` d + e) `wrt` d)
    , TestCase $ assertEqual "matrix composed multiplication gradient" (M.transpose $ unSymbol c) (unSymbol $ getGradientsOf (c `matMul` d `matMul` e) `wrt` d)
    ]
  where
    a = symbol "a" (M.replicate (3, 3) 3.0) :: SymbolMat Float
    b = symbol "b" (M.replicate (3, 3) (-3.0)) :: SymbolMat Float
    c = symbol "c" (M.fromLists (1, 2) [[5.0, 3.0]]) :: SymbolMat Float
    d = symbol "d" (M.fromLists (2, 1) [[1.0], [-2.0]]) :: SymbolMat Float
    e = symbol "e" (M.fromLists (1, 1) [[1.0]]) :: SymbolMat Float


tests :: Test
tests = TestLabel "AutogradTest" $ TestList
    [ testNumOps
    , testNthOrderGradients
    , testVecOps
    , testMatOps
    ]
