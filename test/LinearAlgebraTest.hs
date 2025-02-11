-- | Tests @Synapse.LinearAlgebra@ module and its submodules.


module LinearAlgebraTest
    ( tests
    ) where


import Synapse.LinearAlgebra ((!))

import Synapse.LinearAlgebra.Vec (Vec)
import qualified Synapse.LinearAlgebra.Vec as V

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M

import Test.HUnit


testVecOps :: Test
testVecOps = TestLabel "testVecOps" $ TestList
    [ TestCase $ assertBool "true result == manual operations == linear combination" $
                 all (V.fromList [-1, 2, 5] ==)
                     [vec1 V.*. 2 - vec2, V.linearCombination [(2, vec1), (-1, vec2)]]
    , TestCase $ assertEqual "addition" (V.replicate 3 4) (vec1 + vec2)
    , TestCase $ assertEqual "dot multiplication" 10 (vec1 `V.dot` vec2)
    ]
  where
    vec1 = V.fromList [1, 2, 3] :: Vec Int
    vec2 = V.fromList [3, 2, 1] :: Vec Int

testVecMagnitude :: Test
testVecMagnitude = TestLabel "testVecMagnitude" $ TestList
    [ TestCase $ assertEqual "magnitude value" 5.0 (V.magnitude vec)
    , TestCase $ assertEqual "normalized" (V.fromList [0.6, 0.8]) (V.normalized vec)
    , TestCase $ assertEqual "clamped magnitude" 0.5 (V.magnitude $ V.clampMagnitude 0.5 vec)
    ]
  where
    vec = V.fromList [3.0, 4.0] :: Vec Float

testVecAngle :: Test
testVecAngle = TestLabel "testVecAngle" $ TestList
    [ TestCase $ assertEqual "90-degree angle" (pi / 2.0) (V.angleBetween vec1 vec2)
    , TestCase $ assertEqual "magnitude independent angle" (pi / 2.0) (V.angleBetween (vec1 V.*. 3.0) (vec2 V.*. 4.0))
    , TestCase $ assertEqual "180-degree angle" pi (V.angleBetween vec3 (negate vec3))
    , TestCase $ assertEqual "45-degree angle" (pi / 4.0) (V.angleBetween vec4 vec5)

    , TestCase $ assertEqual "lerp" (V.fromList [0.5, 0.5]) (V.lerp 0.5 vec1 vec2)
    ]
  where
    vec1 = V.fromList [1.0, 0.0] :: Vec Float
    vec2 = V.fromList [0.0, 1.0] :: Vec Float
    vec3 = V.fromList [3.0, 4.0] :: Vec Float
    vec4 = V.fromList [1.0, 0.0] :: Vec Float
    vec5 = V.fromList [1.0, 1.0] :: Vec Float


testMatExtracting :: Test
testMatExtracting = TestLabel "testMatExtracting" $ TestList
    [ TestCase $ assertBool "transposed indexation" $ and [mat ! (r, c) == matT ! (c, r) | r <- [0 .. 2], c <- [0 .. 2]]
    , TestCase $ assertBool "get rows"
                 (  (V.fromList [1, 2, 3] == M.indexRow mat 0)
                 && (V.fromList [4, 5, 6] == M.indexRow mat 1)
                 && (V.fromList [7, 8, 9] == M.indexRow mat 2)
                 )
    , TestCase $ assertBool "get columns"
                 (  (V.fromList [1, 4, 7] == M.indexCol mat 0)
                 && (V.fromList [2, 5, 8] == M.indexCol mat 1)
                 && (V.fromList [3, 6, 9] == M.indexCol mat 2)
                 )
    , TestCase $ assertEqual "transposed transposed" mat (M.transpose matT)
    , TestCase $ assertEqual "diagonal" (V.fromList [1, 5, 9]) (M.diagonal mat)
    ]
  where
    mat = M.fromLists (3, 3) [[1, 2, 3], [4, 5, 6], [7, 8, 9]] :: Mat Int
    matT = M.transpose mat :: Mat Int

testMatOps :: Test
testMatOps = TestLabel "testMatOps" $ TestList
    [ TestCase $ assertEqual "num addition" (M.fromLists (3, 3) [[2, 3, 4], [5, 6, 7], [8, 9, 10]]) (mat1 M.+. 1)
    , TestCase $ assertEqual "num multiplication" (M.fromLists (3, 3) [[2, 4, 6], [8, 10, 12], [14, 16, 18]]) (mat1 M.*. 2)

    , TestCase $ assertEqual "addition" (M.replicate (3, 3) 10) (mat1 + mat2)
    , TestCase $ assertEqual "transposed addition" (M.fromLists (3, 3) [[10, 8, 6], [12, 10, 8], [14, 12, 10]]) (mat1 + M.transpose mat2)

    , TestCase $ assertEqual "adamar multiplication" (M.fromLists (3, 3) [[9, 16, 21], [24, 25, 24], [21, 16, 9]]) (mat1 * mat2)
    , TestCase $ assertEqual "matrix multiplication" (M.fromLists (3, 3) [[30, 24, 18], [84, 69, 54], [138, 114, 90]]) (mat1 `M.matMul` mat2)

    , TestCase $ assertEqual "map" (M.replicate (3, 3) 5) (M.map (const (5 :: Int)) mat1)
    , TestCase $ assertEqual "imap" (M.fromLists (3, 3) [[1, 3, 5], [5, 7, 9], [9, 11, 13]]) (M.imap (\(r, c) x -> r + c + x) mat1)

    , TestCase $ assertEqual "swap rows" (M.fromLists (3, 3) [[7, 8, 9], [4, 5, 6], [1, 2, 3]]) (M.swapRows mat1 0 2)
    , TestCase $ assertEqual "swap cols" (M.fromLists (3, 3) [[2, 1, 3], [5, 4, 6], [8, 7, 9]]) (M.swapCols mat1 0 1)

    , TestCase $ assertEqual "extend" (M.fromLists (4, 4) [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0]]) (M.extend mat1 0 (4, 4))
    , TestCase $ assertEqual "failed extend" mat1 (M.extend mat1 0 (2, 2))
    , TestCase $ assertEqual "shrink" (M.fromLists (2, 2) [[1, 2], [4, 5]]) (M.shrink mat1 (2, 2))
    , TestCase $ assertEqual "failed shrink" mat1 (M.shrink mat1 (4, 4))
    ]
  where
    mat1 = M.fromLists (3, 3) [[1, 2, 3], [4, 5, 6], [7, 8, 9]] :: Mat Int
    mat2 = M.fromLists (3, 3) [[9, 8, 7], [6, 5, 4], [3, 2, 1]] :: Mat Int

testMatSubmatrices :: Test
testMatSubmatrices = TestLabel "testMatSubmatrices" $ TestList
    [ TestCase $ assertEqual "minor" (M.fromLists (2, 2) [[1, 3], [7, 9]]) (M.minor mat1 (1, 1))
    , TestCase $ assertEqual "submatrix" (M.fromLists (2, 2) [[2, 3], [5, 6]]) (M.submatrix mat1 ((0, 1), (2, 3)))
    , TestCase $ assertEqual "split" (M.replicate (4, 4) 1, M.replicate (4, 4) 2, M.replicate (4, 4) 3, M.replicate (4, 4) 4) (M.split mat2 (4, 4))
    , TestCase $ assertEqual "join" (M.fromLists (2, 2) [[1, 2], [3, 4]]) (M.join (matTL, matTR, matBL, matBR))
    , TestCase $ assertEqual "join horizontal" (M.fromLists (1, 2) [[1, 2]]) (matTL M.<|> matTR)
    , TestCase $ assertEqual "join vertical" (M.fromLists (2, 1) [[1], [3]]) (matTL M.<-> matBL)
    ]
  where
    mat1 = M.fromLists (3, 3) [[1, 2, 3], [4, 5, 6], [7, 8, 9]] :: Mat Int
    mat2 = M.generate (8, 8) (\(r, c) -> if r >= 4 && c >= 4 then 4 else if r >= 4 then 3 else if c >= 4 then 2 else 1) :: Mat Int
    matTL = M.singleton 1 :: Mat Int
    matTR = M.singleton 2 :: Mat Int
    matBL = M.singleton 3 :: Mat Int
    matBR = M.singleton 4 :: Mat Int

testMatDetInverse :: Test
testMatDetInverse = TestLabel "testMatDetInverse" $ TestList
    [ TestCase $ assertEqual "det == 0" 0 (M.det mat1)
    , TestCase $ assertEqual "det" (-230) (M.det mat2)
    , TestCase $ assertEqual "rref" (M.fromLists (2, 4) [[1.0, 0.0, -3.0, -4.0], [0.0, 1.0, 1.0, 1.0]]) (M.rref mat3)
    , TestCase $ assertEqual "inverse" (Just $ M.fromLists (2, 2) [[-3.0, -4.0], [1.0, 1.0]]) (M.inverse mat4)
    ]
  where
    mat1 = M.fromLists (3, 3) [[1, 2, 3], [4, 5, 6], [7, 8, 9]] :: Mat Int
    mat2 = M.fromLists (3, 3) [[2, 5, 5], [4, -10, 0], [-3, -2, 1]] :: Mat Int
    mat3 = M.fromLists (2, 4) [[1.0, 4.0, 1.0, 0.0], [-1.0, -3.0, 0.0, 1.0]] :: Mat Float
    mat4 = M.fromLists (2, 2) [[1.0, 4.0], [-1.0, -3.0]] :: Mat Float


tests :: Test
tests = TestLabel "LinearAlgebraTest" $ TestList
    [ -- @Vec@
      testVecOps
    , testVecMagnitude
    , testVecAngle

      -- @Mat@
    , testMatExtracting
    , testMatOps
    , testMatSubmatrices
    , testMatDetInverse
    ]
