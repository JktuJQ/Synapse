{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Full and axis-wise reductions that remain inside the tensor computation.
module Synapse.Tensor.Ops.Reduction
  ( sum,
    sumAxes,
    mean,
    meanAxes,
    normL1,
    normL2,
    minimum,
    minimumAxes,
    maximum,
    maximumAxes,
    argmax,
    argmin,
    all,
    any,
    allAxes,
    anyAxes,
    softmax,
    logSoftmax,
    logSumExp,
    logSumExpAxes,
  )
where

import Data.Proxy (Proxy (..))
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Reduction as Pullback
import Synapse.Tensor.Ops (Needs (Fractional, Numeric), TensorOps (..))
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Primitives.Reduction as Raw
import Synapse.Tensor.Shapes.Axes (Axes, CanReduce, CanReduceNonEmpty, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor, TensorElement)
import Prelude hiding (all, any, maximum, minimum, sum)

-- | Sum all elements, returning zero for an empty tensor.
sum ::
  forall {t} sh a.
  (TensorOps t a, Num (TensorElement a), Shape sh) =>
  t sh a ->
  t '[] a
sum = customVJP Numeric Raw.sum (\primalInput -> let output = Raw.sum primalInput in (output, (primalInput, output))) (uncurry Pullback.sum)

-- | Sum over selected axes and remove them from the output shape.
-- Axis order does not affect the result and an empty axis list preserves the input.
-- Empty reduction regions produce zero.
sumAxes ::
  forall {t} xs sh a.
  (TensorOps t a, Num (TensorElement a), Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
sumAxes = customVJP Numeric (Raw.sumAxes @xs) (\primalInput -> let output = (Raw.sumAxes @xs) primalInput in (output, (primalInput, output))) (uncurry (Pullback.sumAxes @xs))

-- | Mean of all floating point elements, returning NaN for an empty tensor.
mean ::
  forall {t} sh a.
  (TensorOps t a, RealFloat a, Fractional (TensorElement a), Shape sh) =>
  t sh a ->
  t '[] a
mean = customVJP Fractional Raw.mean (\primalInput -> let output = Raw.mean primalInput in (output, (primalInput, output))) (uncurry Pullback.mean)

-- | Mean over selected axes, removing them from the output shape.
-- Empty reduction regions produce NaN and an empty axis list preserves the input.
meanAxes ::
  forall {t} xs sh a.
  (TensorOps t a, RealFloat a, Fractional (TensorElement a), Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
meanAxes = customVJP Fractional (Raw.meanAxes @xs) (\primalInput -> let output = (Raw.meanAxes @xs) primalInput in (output, (primalInput, output))) (uncurry (Pullback.meanAxes @xs))

-- | L1 vector norm over all elements: @sum (abs input)@.
-- Returns a scalar tensor, or zero for an empty tensor.
-- For matrix inputs this treats all entries as one vector, rather than
-- computing an induced matrix norm.
normL1 ::
  forall {t} sh a.
  (TensorOps t a, Num (TensorElement a), Shape sh) =>
  t sh a ->
  t '[] a
normL1 = customVJP Numeric Raw.normL1 (\input -> let output = Raw.normL1 input in (output, (input, output))) (uncurry Pullback.normL1)

-- | L2 vector norm over all elements: @sqrt (sum (input * input))@.
-- Returns a scalar tensor, or zero for an empty tensor. For matrices this
-- is the Frobenius norm. Finite inputs are scaled before squaring to avoid
-- intermediate overflow and underflow. NaN values propagate.
-- Differentiation chooses the zero subgradient at the all-zero input.
normL2 ::
  forall {t} sh a.
  (TensorOps t a, Floating (TensorElement a), E.Ord (TensorElement a), E.RealFloat (TensorElement a), Shape sh) =>
  t sh a ->
  t '[] a
normL2 = customVJP Fractional Raw.normL2 (\input -> let output = Raw.normL2 input in (output, (input, output))) (uncurry Pullback.normL2)

-- | Minimum of all elements of a statically nonempty tensor.
-- Floating point NaN values propagate to the result.
minimum ::
  forall {t} sh a.
  (TensorOps t a, E.Ord (TensorElement a), Shape sh, NonEmpty sh) =>
  t sh a ->
  t '[] a
minimum = customVJP Fractional Raw.minimum (\primalInput -> let output = Raw.minimum primalInput in (output, (primalInput, output))) (uncurry Pullback.extremum)

-- | Minimum over selected axes, removing them from the output shape.
-- Selected dimensions must be nonzero, while unselected dimensions may be empty.
-- Floating point NaN values propagate to the result.
minimumAxes ::
  forall {t} xs sh a.
  (TensorOps t a, E.Ord (TensorElement a), Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
minimumAxes = customVJP Fractional (Raw.minimumAxes @xs) (\primalInput -> let output = (Raw.minimumAxes @xs) primalInput in (output, (primalInput, output))) (uncurry (Pullback.extremumAxes @xs))

-- | Maximum of all elements of a statically nonempty tensor.
-- Floating point NaN values propagate to the result.
maximum ::
  forall {t} sh a.
  (TensorOps t a, E.Ord (TensorElement a), Shape sh, NonEmpty sh) =>
  t sh a ->
  t '[] a
maximum = customVJP Fractional Raw.maximum (\primalInput -> let output = Raw.maximum primalInput in (output, (primalInput, output))) (uncurry Pullback.extremum)

-- | Maximum over selected axes, removing them from the output shape.
-- Selected dimensions must be nonzero, while unselected dimensions may be empty.
-- Floating point NaN values propagate to the result.
maximumAxes ::
  forall {t} xs sh a.
  (TensorOps t a, E.Ord (TensorElement a), Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduceNonEmpty xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
maximumAxes = customVJP Fractional (Raw.maximumAxes @xs) (\primalInput -> let output = (Raw.maximumAxes @xs) primalInput in (output, (primalInput, output))) (uncurry (Pullback.extremumAxes @xs))

-- | Test whether every element is true, returning True for an empty tensor.
all ::
  forall {t} sh.
  (TensorOps t Bool, Shape sh) =>
  t sh Bool ->
  t '[] Bool
all = literal . Raw.all . primal

-- | Test whether any element is true, returning False for an empty tensor.
any ::
  forall {t} sh.
  (TensorOps t Bool, Shape sh) =>
  t sh Bool ->
  t '[] Bool
any = literal . Raw.any . primal

-- | Test conjunction along selected axes, removing them from the result.
-- Empty reduction regions produce True.
allAxes ::
  forall {t} xs sh.
  (TensorOps t Bool, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh Bool ->
  t (ReduceShape xs sh) Bool
allAxes = literal . Raw.allAxes @xs . primal

-- | Test disjunction along selected axes, removing them from the result.
-- Empty reduction regions produce False.
anyAxes ::
  forall {t} xs sh.
  (TensorOps t Bool, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh Bool ->
  t (ReduceShape xs sh) Bool
anyAxes = literal . Raw.anyAxes @xs . primal

-- | Normalize exponentials along one axis without overflowing on large finite inputs.
-- The axis must be nonempty; NaN and infinite logits follow floating point arithmetic.
softmax ::
  forall {t} axis sh a.
  (TensorOps t a, Floating a, Floating (TensorElement a), E.Ord (TensorElement a), Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
  t sh a ->
  t sh a
softmax = customVJP Fractional (Raw.softmax @axis) (\primalInput -> let output = (Raw.softmax @axis) primalInput in (output, (primalInput, output))) (uncurry (Pullback.softmax (Proxy @axis)))

-- | Logarithm of softmax along one nonempty axis, computed without taking the log of small probabilities.
logSoftmax ::
  forall {t} axis sh a.
  (TensorOps t a, Floating a, Floating (TensorElement a), E.Ord (TensorElement a), Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
  t sh a ->
  t sh a
logSoftmax = customVJP Fractional (Raw.logSoftmax @axis) (\primalInput -> let output = (Raw.logSoftmax @axis) primalInput in (output, (primalInput, output))) (uncurry (Pullback.logSoftmax (Proxy @axis)))

-- | Logarithm of the sum of exponentials, using maximum subtraction for finite inputs.
-- Empty tensors produce negative infinity and NaN values propagate.
logSumExp ::
  forall {t} sh a.
  (TensorOps t a, RealFloat a, Floating (TensorElement a), E.Ord (TensorElement a), E.RealFloat (TensorElement a), Shape sh) =>
  t sh a ->
  t '[] a
logSumExp = customVJP Fractional Raw.logSumExp (\primalInput -> let output = Raw.logSumExp primalInput in (output, (primalInput, output))) (uncurry Pullback.logSumExp)

-- | Logarithm of summed exponentials over selected axes, removing those axes.
-- Empty regions produce negative infinity; infinite extrema are preserved and NaN values propagate.
logSumExpAxes ::
  forall {t} xs sh a.
  (TensorOps t a, RealFloat a, Floating (TensorElement a), E.Ord (TensorElement a), E.RealFloat (TensorElement a), Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh) =>
  t sh a ->
  t (ReduceShape xs sh) a
logSumExpAxes = customVJP Fractional (Raw.logSumExpAxes @xs) (\primalInput -> let output = (Raw.logSumExpAxes @xs) primalInput in (output, (primalInput, output))) (uncurry (Pullback.logSumExpAxes @xs))

-- | Position of the maximum along one nonempty axis, removing that axis.
-- Positions start at zero; ties select the first occurrence and NaN selects the first NaN.
-- The result is an ordinary integer tensor, including for differentiable inputs, and carries no gradient history.
-- If the input representation is not fixed by context, annotate the input tensor rather than the result.
-- For example, @argmax \@1 (zeros :: Tensor '[2, 3] Float)@ selects one position per row.
argmax ::
  forall {t} axis sh a.
  (TensorOps t a, E.Ord (TensorElement a), Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int
argmax = Raw.argmax @axis . primal

-- | Position of the minimum along one nonempty axis, removing that axis.
-- Positions start at zero; ties select the first occurrence and NaN selects the first NaN.
-- The result is an ordinary integer tensor, including for differentiable inputs, and carries no gradient history.
-- If the input representation is not fixed by context, annotate the input tensor as for 'argmax'.
argmin ::
  forall {t} axis sh a.
  (TensorOps t a, E.Ord (TensorElement a), Shape sh, Shape (ReduceShape '[axis] sh), Axes '[axis], CanReduceNonEmpty '[axis] sh) =>
  t sh a ->
  Tensor (ReduceShape '[axis] sh) Int
argmin = Raw.argmin @axis . primal
