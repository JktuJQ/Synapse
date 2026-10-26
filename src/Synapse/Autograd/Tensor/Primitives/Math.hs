{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Vector-Jacobian products for arithmetic, elementary functions, and matrix computations.
module Synapse.Autograd.Tensor.Primitives.Math (add, subtract, multiply, negate, abs, signum, divide, recip, exp, log, sqrt, power, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, dot, matmul) where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.Primitives.Elementwise (map)
import Synapse.Tensor.Primitives.Math (broadcastBatchIndex, broadcastCopies, broadcastDimensions, contract, matrixDimensions)
import qualified Synapse.Tensor.Primitives.Math as Raw
import Synapse.Tensor.Shapes.Dimensions (Shape, shape, shapeSize)
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..))
import Prelude hiding (abs, acos, acosh, asin, asinh, atan, atanh, cos, cosh, exp, log, map, negate, recip, signum, sin, sinh, sqrt, subtract, tan, tanh)
import qualified Prelude as P

-- | Send the seed unchanged to both operands.
add :: a -> a -> a -> a -> (a, a)
add _ _ _ seed = (seed, seed)

-- | Negate the contribution to the subtracted operand.
subtract :: (Num a) => a -> a -> a -> a -> (a, a)
subtract _ _ _ seed = (seed, P.negate seed)

-- | Scale each operand's seed by the other operand.
multiply :: (Num a) => a -> a -> a -> a -> (a, a)
multiply x y _ seed = (seed * y, seed * x)

-- | Reverse the sign of the seed.
negate :: (Num a) => a -> a -> a -> a
negate _ _ = P.negate

-- | Use the input sign, including a zero derivative at zero.
abs :: (Num a) => a -> a -> a -> a
abs x _ seed = seed * P.signum x

-- | Discard the seed using a zero with the input's structure.
signum :: (a -> a) -> a -> a -> a -> a
signum zeros x _ _ = zeros x

-- | Differentiate both numerator and denominator.
divide :: (Fractional a) => a -> a -> a -> a -> (a, a)
divide x y _ seed = (seed / y, P.negate (seed * x / (y * y)))

-- | Scale the negated seed by the reciprocal input square.
recip :: (Fractional a) => a -> a -> a -> a
recip x _ seed = P.negate (seed / (x * x))

-- | Reuse the forward exponential when scaling the seed.
exp :: (Num a) => a -> a -> a -> a
exp _ output seed = seed * output

-- | Divide the seed by the input.
log :: (Fractional a) => a -> a -> a -> a
log x _ seed = seed / x

-- | Reuse the forward square root in the denominator.
sqrt :: (Fractional a) => a -> a -> a -> a
sqrt _ output seed = seed / (2 * output)

-- | Differentiate both base and exponent, reusing the forward power for the exponent contribution.
power :: (Floating a) => a -> a -> a -> a -> (a, a)
power x y output seed = (seed * y * x ** (y - 1), seed * output * P.log x)

-- | Scale the seed by the input cosine.
sin :: (Floating a) => a -> a -> a -> a
sin x _ seed = seed * P.cos x

-- | Scale the seed by the negated input sine.
cos :: (Floating a) => a -> a -> a -> a
cos x _ seed = seed * P.negate (P.sin x)

-- | Divide the seed by the squared input cosine.
tan :: (Floating a) => a -> a -> a -> a
tan x _ seed = seed / (P.cos x * P.cos x)

-- | Scale the seed by the derivative of inverse sine.
asin :: (Floating a) => a -> a -> a -> a
asin x _ seed = seed / P.sqrt (1 - x * x)

-- | Scale the seed by the derivative of inverse cosine.
acos :: (Floating a) => a -> a -> a -> a
acos x _ seed = seed / P.negate (P.sqrt (1 - x * x))

-- | Scale the seed by the derivative of inverse tangent.
atan :: (Floating a) => a -> a -> a -> a
atan x _ seed = seed / (1 + x * x)

-- | Scale the seed by the input hyperbolic cosine.
sinh :: (Floating a) => a -> a -> a -> a
sinh x _ seed = seed * P.cosh x

-- | Scale the seed by the input hyperbolic sine.
cosh :: (Floating a) => a -> a -> a -> a
cosh x _ seed = seed * P.sinh x

-- | Reuse the forward hyperbolic tangent to compute its derivative.
tanh :: (Num a) => a -> a -> a -> a
tanh _ output seed = seed * (1 - output * output)

-- | Scale the seed by the derivative of inverse hyperbolic sine.
asinh :: (Floating a) => a -> a -> a -> a
asinh x _ seed = seed / P.sqrt (1 + x * x)

-- | Scale the seed by the derivative of inverse hyperbolic cosine.
acosh :: (Floating a) => a -> a -> a -> a
acosh x _ seed = seed / (P.sqrt (x - 1) * P.sqrt (x + 1))

-- | Scale the seed by the derivative of inverse hyperbolic tangent.
atanh :: (Fractional a) => a -> a -> a -> a
atanh x _ seed = seed / (1 - x * x)

-- | Scale each vector by the other vector and the scalar output seed.
dot :: (Shape '[n], DType a, Num (TensorElement a)) => Tensor '[n] a -> Tensor '[n] a -> Tensor '[] a -> Tensor '[] a -> (Tensor '[n] a, Tensor '[n] a)
dot x y _ (Tensor gradient) =
  let scale = TensorElement (A.the gradient)
   in (map (* scale) y, map (* scale) x)

-- | Contract directly into each original input element, summing all broadcast batch copies.
matmul :: forall lhs rhs out a. (Shape lhs, Shape rhs, Shape out, DType a, Num (TensorElement a)) => Tensor lhs a -> Tensor rhs a -> Tensor out a -> Tensor out a -> (Tensor lhs a, Tensor rhs a)
matmul (Tensor left) (Tensor right) _ (Tensor gradient) =
  ( Tensor (A.reshape (A.shape left) leftGradient),
    Tensor (A.reshape (A.shape right) rightGradient)
  )
  where
    (leftBatch, rows, shared) = matrixDimensions True (shape @lhs)
    (rightBatch, _, cols) = matrixDimensions False (shape @rhs)
    batchDims = broadcastDimensions leftBatch rightBatch
    (leftCopies, expandLeft) = broadcastCopies leftBatch batchDims
    (rightCopies, expandRight) = broadcastCopies rightBatch batchDims
    leftVector = A.flatten left
    rightVector = A.flatten right
    outputVector = A.flatten gradient
    leftGradient = contract (shapeSize @lhs) (leftCopies * cols) $ \linear entry ->
      let rowIndex = linear `A.quot` A.constant shared
          inner = linear `A.rem` A.constant shared
          row = rowIndex `A.rem` A.constant rows
          originalBatch = rowIndex `A.quot` A.constant rows
          batch = expandLeft originalBatch (entry `A.quot` A.constant cols)
          col = entry `A.rem` A.constant cols
          outputOffset = (batch * A.constant rows + row) * A.constant cols + col
          rightOffset = (broadcastBatchIndex rightBatch batchDims batch * A.constant shared + inner) * A.constant cols + col
       in Raw.multiply (outputVector A.! A.index1 outputOffset) (rightVector A.! A.index1 rightOffset)
    rightGradient = contract (shapeSize @rhs) (rightCopies * rows) $ \linear entry ->
      let col = linear `A.rem` A.constant cols
          innerIndex = linear `A.quot` A.constant cols
          inner = innerIndex `A.rem` A.constant shared
          originalBatch = innerIndex `A.quot` A.constant shared
          batch = expandRight originalBatch (entry `A.quot` A.constant rows)
          row = entry `A.rem` A.constant rows
          leftOffset = (broadcastBatchIndex leftBatch batchDims batch * A.constant rows + row) * A.constant shared + inner
          outputOffset = (batch * A.constant rows + row) * A.constant cols + col
       in Raw.multiply (leftVector A.! A.index1 leftOffset) (outputVector A.! A.index1 outputOffset)
