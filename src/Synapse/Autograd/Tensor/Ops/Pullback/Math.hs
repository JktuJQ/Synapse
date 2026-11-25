-- | Vector-Jacobian products for arithmetic, elementary functions, and matrix computations.
module Synapse.Autograd.Tensor.Ops.Pullback.Math
    ( -- * Arithmetic
      add
    , subtract
    , multiply
    , negate
    , abs
    , signum
    , divide
    , recip

      -- * Exponentials and powers
    , exp
    , log
    , sqrt
    , power

      -- * Trigonometric functions
    , sin
    , cos
    , tan
    , asin
    , acos
    , atan

      -- * Hyperbolic functions
    , sinh
    , cosh
    , tanh
    , asinh
    , acosh
    , atanh

      -- * Products
    , dot
    , matmulWith
    ) where

import Prelude hiding
    ( abs
    , acos
    , acosh
    , asin
    , asinh
    , atan
    , atanh
    , cos
    , cosh
    , exp
    , log
    , map
    , negate
    , recip
    , signum
    , sin
    , sinh
    , sqrt
    , subtract
    , tan
    , tanh
    )

import Synapse.Tensor.Ops.Constraints (NumDType)
import Synapse.Tensor.Ops.Raw.Elementwise (map)
import Synapse.Tensor.Ops.Raw.Math
    ( broadcastBatchIndex
    , broadcastCopies
    , broadcastDimensions
    , contract
    , matrixDimensions
    )
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Dimensions (Shape, shape)

import Data.Array.Accelerate qualified as A
import Prelude qualified as P

import Synapse.Tensor.Ops.Raw.Math qualified as Raw

-- | Send the seed unchanged to both operands.
add
    :: a
    -> a
    -> a
    -> a
    -> (a, a)
add _ _ _ seed = (seed, seed)

-- | Negate the contribution to the subtracted operand.
subtract
    :: Num a
    => a
    -> a
    -> a
    -> a
    -> (a, a)
subtract _ _ _ seed = (seed, P.negate seed)

-- | Scale each operand's seed by the other operand.
multiply
    :: Num a
    => a
    -> a
    -> a
    -> a
    -> (a, a)
multiply x y _ seed = (seed * y, seed * x)

-- | Reverse the sign of the seed.
negate
    :: Num a
    => a
    -> a
    -> a
    -> a
negate _ _ = P.negate

-- | Use the input sign, including a zero derivative at zero.
abs
    :: Num a
    => a
    -> a
    -> a
    -> a
abs x _ seed = seed * P.signum x

-- | Discard the seed using a zero with the input's structure.
signum
    :: a
    -> a
    -> a
    -> a
    -> a
signum zeros _ _ _ = zeros

-- | Differentiate both numerator and denominator.
divide
    :: Fractional a
    => a
    -> a
    -> a
    -> a
    -> (a, a)
divide x y _ seed = (seed / y, P.negate (seed * x / (y * y)))

-- | Scale the negated seed by the reciprocal input square.
recip
    :: Fractional a
    => a
    -> a
    -> a
    -> a
recip x _ seed = P.negate (seed / (x * x))

-- | Reuse the forward exponential when scaling the seed.
exp
    :: Num a
    => a
    -> a
    -> a
    -> a
exp _ output seed = seed * output

-- | Divide the seed by the input.
log
    :: Fractional a
    => a
    -> a
    -> a
    -> a
log x _ seed = seed / x

-- | Reuse the forward square root in the denominator.
sqrt
    :: Fractional a
    => a
    -> a
    -> a
    -> a
sqrt _ output seed = seed / (2 * output)

-- | Differentiate both base and exponent, reusing the forward power for the exponent contribution.
power
    :: Floating a
    => a
    -> a
    -> a
    -> a
    -> (a, a)
power x y output seed = (seed * y * x ** (y - 1), seed * output * P.log x)

-- | Scale the seed by the input cosine.
sin
    :: Floating a
    => a
    -> a
    -> a
    -> a
sin x _ seed = seed * P.cos x

-- | Scale the seed by the negated input sine.
cos
    :: Floating a
    => a
    -> a
    -> a
    -> a
cos x _ seed = seed * P.negate (P.sin x)

-- | Divide the seed by the squared input cosine.
tan
    :: Floating a
    => a
    -> a
    -> a
    -> a
tan x _ seed = seed / (P.cos x * P.cos x)

-- | Scale the seed by the derivative of inverse sine.
asin
    :: Floating a
    => a
    -> a
    -> a
    -> a
asin x _ seed = seed / P.sqrt (1 - x * x)

-- | Scale the seed by the derivative of inverse cosine.
acos
    :: Floating a
    => a
    -> a
    -> a
    -> a
acos x _ seed = seed / P.negate (P.sqrt (1 - x * x))

-- | Scale the seed by the derivative of inverse tangent.
atan
    :: Floating a
    => a
    -> a
    -> a
    -> a
atan x _ seed = seed / (1 + x * x)

-- | Scale the seed by the input hyperbolic cosine.
sinh
    :: Floating a
    => a
    -> a
    -> a
    -> a
sinh x _ seed = seed * P.cosh x

-- | Scale the seed by the input hyperbolic sine.
cosh
    :: Floating a
    => a
    -> a
    -> a
    -> a
cosh x _ seed = seed * P.sinh x

-- | Reuse the forward hyperbolic tangent to compute its derivative.
tanh
    :: Num a
    => a
    -> a
    -> a
    -> a
tanh _ output seed = seed * (1 - output * output)

-- | Scale the seed by the derivative of inverse hyperbolic sine.
asinh
    :: Floating a
    => a
    -> a
    -> a
    -> a
asinh x _ seed = seed / P.sqrt (1 + x * x)

-- | Scale the seed by the derivative of inverse hyperbolic cosine.
acosh
    :: Floating a
    => a
    -> a
    -> a
    -> a
acosh x _ seed = seed / (P.sqrt (x - 1) * P.sqrt (x + 1))

-- | Scale the seed by the derivative of inverse hyperbolic tangent.
atanh
    :: Fractional a
    => a
    -> a
    -> a
    -> a
atanh x _ seed = seed / (1 - x * x)

-- | Scale each vector by the other vector and the scalar output seed.
dot
    :: (Shape '[n], NumDType a)
    => Tensor '[n] a
    -> Tensor '[n] a
    -> Tensor '[] a
    -> Tensor '[] a
    -> (Tensor '[n] a, Tensor '[n] a)
dot x y _ (Tensor gradient) =
    let scale = TensorElement (A.the gradient)
     in (map (* scale) y, map (* scale) x)

-- | A flattened matrix operand with its batch shape and transpose flag.
type MatrixOperand a = ([Int], Bool, A.Acc (A.Vector a))

-- | Read an element from a broadcast matrix in its logical orientation.
matrixElement
    :: DType a
    => [Int]
    -> Int
    -> Int
    -> MatrixOperand a
    -> A.Exp Int
    -> A.Exp Int
    -> A.Exp Int
    -> A.Exp a
matrixElement batchDims rows cols (sourceBatch, transposed, values) batch row col =
    values A.! A.index1 (batchOffset + offset)
  where
    batchOffset = broadcastBatchIndex sourceBatch batchDims batch * A.constant (rows * cols)
    offset
        | transposed = col * A.constant rows + row
        | otherwise = row * A.constant cols + col

-- | Multiply matrices and sum any batch axes broadcast by the differentiated operand.
gradientProduct
    :: NumDType a
    => [Int]
    -> [Int]
    -> (Int, Int, Int)
    -> MatrixOperand a
    -> MatrixOperand a
    -> A.Acc (A.Vector a)
gradientProduct batchDims targetBatch (rows, shared, cols) left@(lb, lt, xs) right@(rb, rt, ys)
    | copies == 1 = Raw.matrixProduct lb rb batchDims rows shared cols lt rt xs ys
    | otherwise = contract (product targetBatch * rows * cols) (copies * shared) $ \linear entry ->
        let (matrixRow, col) = A.quotRem linear (A.constant cols)
            (originalBatch, row) = A.quotRem matrixRow (A.constant rows)
            (copy, inner) = A.quotRem entry (A.constant shared)
            batch = expand originalBatch copy
         in Raw.multiply
                (matrixElement batchDims rows shared left batch row inner)
                (matrixElement batchDims shared cols right batch inner col)
  where
    (copies, expand) = broadcastCopies targetBatch batchDims

-- | Multiply the seed by the opposite operand's transpose for each input gradient.
matmulWith
    :: forall lhs rhs out a
     . (Shape lhs, Shape rhs, Shape out, NumDType a)
    => Bool
    -> Bool
    -> Tensor lhs a
    -> Tensor rhs a
    -> Tensor out a
    -> Tensor out a
    -> (Tensor lhs a, Tensor rhs a)
matmulWith transposeLeft transposeRight (Tensor left) (Tensor right) _ (Tensor gradient) =
    ( Tensor $ A.reshape (A.shape left) leftGradient
    , Tensor $ A.reshape (A.shape right) rightGradient
    )
  where
    (leftBatch, rows, shared) = matrixDimensions True transposeLeft (shape @lhs)
    (rightBatch, _, cols) = matrixDimensions False transposeRight (shape @rhs)
    batchDims = broadcastDimensions leftBatch rightBatch
    leftVector = A.flatten left
    rightVector = A.flatten right
    seed = A.flatten gradient

    leftGradient
        | transposeLeft =
            gradientProduct
                batchDims
                leftBatch
                (shared, cols, rows)
                (rightBatch, transposeRight, rightVector)
                (batchDims, True, seed)
        | otherwise =
            gradientProduct
                batchDims
                leftBatch
                (rows, cols, shared)
                (batchDims, False, seed)
                (rightBatch, not transposeRight, rightVector)

    (productBatch, productLeftBatch, productRightBatch, productRows)
        | not transposeLeft
        , leftBatch == batchDims
        , product rightBatch == 1 =
            ([], [], [], product batchDims * rows)
        | otherwise = (batchDims, leftBatch, rightBatch, rows)

    rightGradient
        | transposeRight =
            gradientProduct
                productBatch
                productRightBatch
                (cols, productRows, shared)
                (productBatch, True, seed)
                (productLeftBatch, transposeLeft, leftVector)
        | otherwise =
            gradientProduct
                productBatch
                productRightBatch
                (shared, productRows, cols)
                (productLeftBatch, not transposeLeft, leftVector)
                (productBatch, False, seed)
