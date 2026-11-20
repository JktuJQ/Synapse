{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

{- | Sums, means, extrema, norms, and exponential normalization.
Masked operations take a boolean tensor of the same shape; True selects an element.
Masked statistics require enough selected elements for their correction.
Masked extrema use their fallback to fill unselected elements.
Choose it beyond selected values; equal values can make an index point to a masked element.
-}
module Synapse.Tensor.Ops.Reduction
    ( -- * Sums and means
      sum
    , sumAxes
    , maskedSum
    , maskedSumAxes
    , mean
    , meanAxes
    , maskedMean
    , maskedMeanAxes
    , variance
    , varianceAxes
    , maskedVariance
    , maskedVarianceAxes
    , stddev
    , stddevAxes
    , maskedStddev
    , maskedStddevAxes

      -- * Norms
    , normL1
    , maskedNormL1
    , normL2
    , maskedNormL2

      -- * Extrema and their positions
    , minimum
    , minimumAxes
    , maskedMinimum
    , maskedMinimumAxes
    , argmin
    , maskedArgmin
    , maximum
    , maximumAxes
    , maskedMaximum
    , maskedMaximumAxes
    , argmax
    , maskedArgmax

      -- * Boolean reductions
    , all
    , allAxes
    , maskedAll
    , maskedAllAxes
    , any
    , anyAxes
    , maskedAny
    , maskedAnyAxes

      -- * Exponential reductions and normalization
    , logSumExp
    , logSumExpAxes
    , maskedLogSumExp
    , maskedLogSumExpAxes
    , softmax
    , maskedSoftmax
    , logSoftmax
    , maskedLogSoftmax
    ) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import qualified Synapse.Autograd.Tensor.Ops.Pullback.Reduction as Pullback
import Synapse.Tensor.Ops.Constraints (FloatingDType, FractionalDType, NumDType, OrdDType, RealFloatDType)
import qualified Synapse.Tensor.Ops.Construction as C
import Synapse.Tensor.Ops.Definition (TensorOps (..), withVJP)
import qualified Synapse.Tensor.Ops.Logic as E
import Synapse.Tensor.Ops.Math ()
import qualified Synapse.Tensor.Ops.Raw.Reduction as Raw
import qualified Synapse.Tensor.Ops.Shape as Shape
import Synapse.Tensor.Representation (Tensor, TensorElement, TensorLike)
import Synapse.Tensor.Shapes.Axes (Axes, CanReduce, CanReduceNonEmpty, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Prelude hiding (all, any, maximum, minimum, sum)

-- | Sum all elements, returning zero for an empty tensor.
sum
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, NumDType a, Shape sh)
    => t sh a
    -> t '[] a
sum =
    withVJP
        (Proxy @(() :: Constraint))
        Raw.sum
        Pullback.sum

-- | Sum over selected axes and remove them from the output shape.
sumAxes
    :: forall {t :: TensorLike} xs sh a
     . (TensorOps t a, NumDType a, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => t sh a
    -> t (ReduceShape xs sh) a
sumAxes =
    withVJP
        (Proxy @(() :: Constraint))
        (Raw.sumAxes @xs)
        (Pullback.sumAxes @xs)

-- | Sum elements selected by a mask.
maskedSum
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , NumDType a
       , Shape sh
       , Num (t sh a)
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedSum mask values = sum $ E.select mask values 0

-- | Sum selected elements over the specified axes.
maskedSumAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , NumDType a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , Num (t sh a)
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t (ReduceShape xs sh) a
maskedSumAxes mask values = sumAxes @xs $ E.select mask values 0

-- | Mean of all floating point elements, returning NaN for an empty tensor.
mean
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, RealFloat a, FractionalDType a, Shape sh)
    => t sh a
    -> t '[] a
mean =
    withVJP
        (Proxy @(() :: Constraint))
        Raw.mean
        Pullback.mean

-- | Compute means over the selected axes.
meanAxes
    :: forall {t :: TensorLike} xs sh a
     . (TensorOps t a, RealFloat a, FractionalDType a, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => t sh a
    -> t (ReduceShape xs sh) a
meanAxes =
    withVJP
        (Proxy @(() :: Constraint))
        (Raw.meanAxes @xs)
        (Pullback.meanAxes @xs)

-- | Mean over selected elements.
maskedMean
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , RealFloat a
       , FractionalDType a
       , Shape sh
       , Num (t sh a)
       , Fractional (t '[] a)
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedMean mask input =
    maskedSum mask input / fromTensor (Raw.sum $ E.select mask 1 0)

-- | Mean over selected elements.
maskedMeanAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , RealFloat a
       , FractionalDType a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , Num (t sh a)
       , Fractional (t (ReduceShape xs sh) a)
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t (ReduceShape xs sh) a
maskedMeanAxes mask input =
    maskedSumAxes @xs mask input
        / fromTensor (Raw.sumAxes @xs $ E.select mask 1 0)

-- | Variance with correction; each region must contain more elements than the correction.
variance
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, RealFloat a, FractionalDType a, Shape sh, Num (t sh a), Fractional (t '[] a))
    => Int
    -> t sh a
    -> t '[] a
variance correction input =
    let centered = input - Shape.broadcast @sh (mean input)
     in sum (centered * centered) / fromIntegral (shapeSize @sh - correction)

-- | Variance with correction; each region must contain more elements than the correction.
varianceAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , RealFloat a
       , FractionalDType a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , Num (t sh a)
       , Fractional (t (ReduceShape xs sh) a)
       )
    => Int
    -> t sh a
    -> t (ReduceShape xs sh) a
varianceAxes correction input =
    let centered = input - Shape.expandReduced @xs input (meanAxes @xs input)
     in sumAxes @xs (centered * centered) / fromIntegral (Raw.reductionSize @xs @sh - correction)

-- | Variance over selected elements.
maskedVariance
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , RealFloat a
       , FractionalDType a
       , Shape sh
       , Num (t sh a)
       , Fractional (t '[] a)
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Int
    -> Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedVariance correction mask input =
    let meanValue = maskedMean mask input
        centered = E.select mask input 0 - Shape.broadcast @sh meanValue
        count = fromTensor $ Raw.sum $ E.select mask 1 0
     in maskedSum mask (centered * centered) / (count - fromIntegral correction)

-- | Variance over selected elements.
maskedVarianceAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , RealFloat a
       , FractionalDType a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , Num (t sh a)
       , Fractional (t (ReduceShape xs sh) a)
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Int
    -> Tensor sh Bool
    -> t sh a
    -> t (ReduceShape xs sh) a
maskedVarianceAxes correction mask input =
    let meanValue = maskedMeanAxes @xs mask input
        centered = E.select mask input 0 - Shape.expandReduced @xs input meanValue
        count = fromTensor $ Raw.sumAxes @xs $ E.select mask 1 0
     in maskedSumAxes @xs mask (centered * centered) / (count - fromIntegral correction)

-- | Standard deviation with correction; each region must contain more elements than the correction.
stddev
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, RealFloat a, FractionalDType a, Shape sh, Num (t sh a), Floating (t '[] a))
    => Int
    -> t sh a
    -> t '[] a
stddev correction = sqrt . variance correction

-- | Standard deviation with correction; each region must contain more elements than the correction.
stddevAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , RealFloat a
       , FractionalDType a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , Num (t sh a)
       , Floating (t (ReduceShape xs sh) a)
       )
    => Int
    -> t sh a
    -> t (ReduceShape xs sh) a
stddevAxes correction = sqrt . varianceAxes @xs correction

-- | Stddev over selected elements.
maskedStddev
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , RealFloat a
       , FractionalDType a
       , Shape sh
       , Num (t sh a)
       , Floating (t '[] a)
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Int
    -> Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedStddev correction mask = sqrt . maskedVariance correction mask

-- | Stddev over selected elements.
maskedStddevAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , RealFloat a
       , FractionalDType a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , Num (t sh a)
       , Floating (t (ReduceShape xs sh) a)
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Int
    -> Tensor sh Bool
    -> t sh a
    -> t (ReduceShape xs sh) a
maskedStddevAxes correction mask = sqrt . maskedVarianceAxes @xs correction mask

-- | Sum the absolute values of all elements.
normL1
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, BackwardConstraints t (Num a), NumDType a, Shape sh)
    => t sh a
    -> t '[] a
normL1 =
    withVJP
        (Proxy @(Num a))
        Raw.normL1
        Pullback.normL1

-- | L1 norm of selected elements; empty selections return zero.
maskedNormL1
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , Shape sh
       , Num (t sh a)
       , BackwardConstraints t (Num a)
       , NumDType a
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedNormL1 mask input = normL1 $ E.select mask input 0

-- | Compute the square root of the sum of squared elements.
normL2
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, BackwardConstraints t (Fractional a), RealFloatDType a, Shape sh)
    => t sh a
    -> t '[] a
normL2 =
    withVJP
        (Proxy @(Fractional a))
        Raw.normL2
        Pullback.normL2

-- | L2 norm of selected elements; empty selections return zero.
maskedNormL2
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , Shape sh
       , Num (t sh a)
       , BackwardConstraints t (Fractional a)
       , RealFloatDType a
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedNormL2 mask input = normL2 $ E.select mask input 0

-- | Minimum of all elements of a statically nonempty tensor.
minimum
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , OrdDType a
       , Shape sh
       , NonEmpty sh
       )
    => t sh a
    -> t '[] a
minimum =
    withVJP
        (Proxy @(Fractional a, Fractional (TensorElement a)))
        Raw.minimum
        Pullback.extremum

-- | Minimum over selected axes, removing them from the output shape.
minimumAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , OrdDType a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduceNonEmpty xs sh
       )
    => t sh a
    -> t (ReduceShape xs sh) a
minimumAxes =
    withVJP
        (Proxy @(Fractional a, Fractional (TensorElement a)))
        (Raw.minimumAxes @xs)
        (Pullback.extremumAxes @xs)

-- | Masked Minimum, using the fallback for masked elements.
maskedMinimum
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , Shape sh
       , NumDType a
       , OrdDType a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , NonEmpty sh
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => a
    -> Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedMinimum fallback mask input = minimum $ E.select mask input (C.full fallback)

-- | Masked Minimum, using the fallback for masked elements.
maskedMinimumAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , Shape sh
       , NumDType a
       , OrdDType a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , CanReduceNonEmpty xs sh
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => a
    -> Tensor sh Bool
    -> t sh a
    -> t (ReduceShape xs sh) a
maskedMinimumAxes fallback mask input = minimumAxes @xs $ E.select mask input (C.full fallback)

-- | Position of the minimum along one nonempty axis, removing that axis.
argmin
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , OrdDType a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
argmin = Raw.argmin @axis . toTensor

-- | Position of the extremum after filling masked elements with the fallback.
maskedArgmin
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduce '[axis] sh
       , OrdDType a
       , CanReduceNonEmpty '[axis] sh
       )
    => a
    -> Tensor sh Bool
    -> t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
maskedArgmin fallback mask input = argmin @axis $ E.select mask (toTensor input) (C.full fallback)

-- | Maximum of all elements of a statically nonempty tensor.
maximum
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , OrdDType a
       , Shape sh
       , NonEmpty sh
       )
    => t sh a
    -> t '[] a
maximum =
    withVJP
        (Proxy @(Fractional a, Fractional (TensorElement a)))
        Raw.maximum
        Pullback.extremum

-- | Maximum over selected axes, removing them from the output shape.
maximumAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , OrdDType a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduceNonEmpty xs sh
       )
    => t sh a
    -> t (ReduceShape xs sh) a
maximumAxes =
    withVJP
        (Proxy @(Fractional a, Fractional (TensorElement a)))
        (Raw.maximumAxes @xs)
        (Pullback.extremumAxes @xs)

-- | Masked Maximum, using the fallback for masked elements.
maskedMaximum
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , Shape sh
       , NumDType a
       , OrdDType a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , NonEmpty sh
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => a
    -> Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedMaximum fallback mask input = maximum $ E.select mask input (C.full fallback)

-- | Masked Maximum, using the fallback for masked elements.
maskedMaximumAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , Shape sh
       , NumDType a
       , OrdDType a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , CanReduceNonEmpty xs sh
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => a
    -> Tensor sh Bool
    -> t sh a
    -> t (ReduceShape xs sh) a
maskedMaximumAxes fallback mask input = maximumAxes @xs $ E.select mask input (C.full fallback)

-- | Position of the maximum along one nonempty axis, removing that axis.
argmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , OrdDType a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
argmax = Raw.argmax @axis . toTensor

-- | Position of the extremum after filling masked elements with the fallback.
maskedArgmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduce '[axis] sh
       , OrdDType a
       , CanReduceNonEmpty '[axis] sh
       )
    => a
    -> Tensor sh Bool
    -> t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
maskedArgmax fallback mask input = argmax @axis $ E.select mask (toTensor input) (C.full fallback)

-- | Test whether every element is true, returning True for an empty tensor.
all
    :: forall {t :: TensorLike} sh
     . (TensorOps t Bool, Shape sh)
    => t sh Bool
    -> t '[] Bool
all = fromTensor . Raw.all . toTensor

-- | Test conjunction along selected axes, removing them from the result.
allAxes
    :: forall {t :: TensorLike} xs sh
     . (TensorOps t Bool, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => t sh Bool
    -> t (ReduceShape xs sh) Bool
allAxes = fromTensor . Raw.allAxes @xs . toTensor

-- | All over selected boolean values; empty regions return True.
maskedAll
    :: forall {t :: TensorLike} sh
     . (TensorOps t Bool, Shape sh)
    => Tensor sh Bool
    -> t sh Bool
    -> t '[] Bool
maskedAll mask = fromTensor . Raw.all . (\values -> E.select mask (toTensor values) (C.full True))

-- | All over selected boolean values; empty regions return True.
maskedAllAxes
    :: forall {t :: TensorLike} xs sh
     . (TensorOps t Bool, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => Tensor sh Bool
    -> t sh Bool
    -> t (ReduceShape xs sh) Bool
maskedAllAxes mask = fromTensor . Raw.allAxes @xs . (\values -> E.select mask (toTensor values) (C.full True))

-- | Test whether any element is true, returning False for an empty tensor.
any
    :: forall {t :: TensorLike} sh
     . (TensorOps t Bool, Shape sh)
    => t sh Bool
    -> t '[] Bool
any = fromTensor . Raw.any . toTensor

-- | Test disjunction along selected axes, removing them from the result.
anyAxes
    :: forall {t :: TensorLike} xs sh
     . (TensorOps t Bool, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => t sh Bool
    -> t (ReduceShape xs sh) Bool
anyAxes = fromTensor . Raw.anyAxes @xs . toTensor

-- | Any over selected boolean values; empty regions return False.
maskedAny
    :: forall {t :: TensorLike} sh
     . (TensorOps t Bool, Shape sh)
    => Tensor sh Bool
    -> t sh Bool
    -> t '[] Bool
maskedAny mask = fromTensor . Raw.any . (\values -> E.select mask (toTensor values) (C.full False))

-- | Any over selected boolean values; empty regions return False.
maskedAnyAxes
    :: forall {t :: TensorLike} xs sh
     . (TensorOps t Bool, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => Tensor sh Bool
    -> t sh Bool
    -> t (ReduceShape xs sh) Bool
maskedAnyAxes mask = fromTensor . Raw.anyAxes @xs . (\values -> E.select mask (toTensor values) (C.full False))

-- | Logarithm of the sum of exponentials, using maximum subtraction for finite inputs.
logSumExp
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, RealFloatDType a, Shape sh)
    => t sh a
    -> t '[] a
logSumExp =
    withVJP
        (Proxy @(() :: Constraint))
        Raw.logSumExp
        Pullback.logSumExp

-- | Logarithm of summed exponentials over selected axes, removing those axes.
logSumExpAxes
    :: forall {t :: TensorLike} xs sh a
     . (TensorOps t a, RealFloatDType a, Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => t sh a
    -> t (ReduceShape xs sh) a
logSumExpAxes =
    withVJP
        (Proxy @(() :: Constraint))
        (Raw.logSumExpAxes @xs)
        (Pullback.logSumExpAxes @xs)

-- | Masked LogSumExp; empty selections return negative infinity.
maskedLogSumExp
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh, RealFloatDType a, E.Conditional (t sh a), E.Condition (t sh a) ~ Tensor sh Bool)
    => Tensor sh Bool
    -> t sh a
    -> t '[] a
maskedLogSumExp mask input = logSumExp $ E.select mask input (C.full $ negate $ 1 / 0)

-- | Masked LogSumExp; empty selections return negative infinity.
maskedLogSumExpAxes
    :: forall {t :: TensorLike} xs sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , RealFloatDType a
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t (ReduceShape xs sh) a
maskedLogSumExpAxes mask input = logSumExpAxes @xs $ E.select mask input (C.full $ negate $ 1 / 0)

-- | Normalize exponentials along one axis without overflowing on large finite inputs.
softmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , FloatingDType a
       , OrdDType a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       )
    => t sh a
    -> t sh a
softmax =
    withVJP
        (Proxy @(() :: Constraint))
        (Raw.softmax @axis)
        (Pullback.softmax (Proxy @axis))

-- | Normalize selected scores; every reduction region must contain a selected element.
maskedSoftmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduce '[axis] sh
       , RealFloat a
       , FloatingDType a
       , OrdDType a
       , CanReduceNonEmpty '[axis] sh
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t sh a
maskedSoftmax mask input = softmax @axis $ E.select mask input (C.full $ negate $ 1 / 0)

-- | Logarithm of softmax along one nonempty axis, computed without taking the log of small probabilities.
logSoftmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , FloatingDType a
       , OrdDType a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       )
    => t sh a
    -> t sh a
logSoftmax =
    withVJP
        (Proxy @(() :: Constraint))
        (Raw.logSoftmax @axis)
        (Pullback.logSoftmax (Proxy @axis))

-- | Log-normalize selected scores; every reduction region must contain a selected element.
maskedLogSoftmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduce '[axis] sh
       , RealFloat a
       , FloatingDType a
       , OrdDType a
       , CanReduceNonEmpty '[axis] sh
       , E.Conditional (t sh a)
       , E.Condition (t sh a) ~ Tensor sh Bool
       )
    => Tensor sh Bool
    -> t sh a
    -> t sh a
maskedLogSoftmax mask input = logSoftmax @axis $ E.select mask input (C.full $ negate $ 1 / 0)
