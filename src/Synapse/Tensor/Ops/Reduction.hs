{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Sums, means, extrema, norms, and exponential normalization.
module Synapse.Tensor.Ops.Reduction
    ( -- * Sums and means
      sum
    , sumAxes
    , mean
    , meanAxes

      -- * Norms
    , normL1
    , normL2

      -- * Extrema and their positions
    , minimum
    , minimumAxes
    , argmin
    , maximum
    , maximumAxes
    , argmax

      -- * Boolean reductions
    , all
    , allAxes
    , any
    , anyAxes

      -- * Exponential reductions and normalization
    , logSumExp
    , logSumExpAxes
    , softmax
    , logSoftmax
    ) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import qualified Synapse.Autograd.Tensor.Ops.Pullback.Reduction as Pullback
import Synapse.Tensor.Ops.Definition (TensorOps (..), withVJP)
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Raw.Reduction as Raw
import Synapse.Tensor.Representation (Tensor, TensorElement, TensorLike)
import Synapse.Tensor.Shapes.Axes (Axes, CanReduce, CanReduceNonEmpty, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Prelude hiding (all, any, maximum, minimum, sum)

-- | Sum all elements, returning zero for an empty tensor.
sum
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Num (TensorElement a), Shape sh)
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
     . (TensorOps t a, Num (TensorElement a), Shape sh, Shape (ReduceShape xs sh), Axes xs, CanReduce xs sh)
    => t sh a
    -> t (ReduceShape xs sh) a
sumAxes =
    withVJP
        (Proxy @(() :: Constraint))
        (Raw.sumAxes @xs)
        (Pullback.sumAxes @xs)

-- | Mean of all floating point elements, returning NaN for an empty tensor.
mean
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, RealFloat a, Fractional (TensorElement a), Shape sh)
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
     . ( TensorOps t a
       , RealFloat a
       , Fractional (TensorElement a)
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       )
    => t sh a
    -> t (ReduceShape xs sh) a
meanAxes =
    withVJP
        (Proxy @(() :: Constraint))
        (Raw.meanAxes @xs)
        (Pullback.meanAxes @xs)

-- | Sum the absolute values of all elements.
normL1
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, BackwardConstraints t (Num a), Num (TensorElement a), Shape sh)
    => t sh a
    -> t '[] a
normL1 =
    withVJP
        (Proxy @(Num a))
        Raw.normL1
        Pullback.normL1

-- | Compute the square root of the sum of squared elements.
normL2
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , BackwardConstraints t (Fractional a)
       , Floating (TensorElement a)
       , E.Ord (TensorElement a)
       , E.RealFloat (TensorElement a)
       , Shape sh
       )
    => t sh a
    -> t '[] a
normL2 =
    withVJP
        (Proxy @(Fractional a))
        Raw.normL2
        Pullback.normL2

-- | Minimum of all elements of a statically nonempty tensor.
minimum
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , E.Ord (TensorElement a)
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
       , E.Ord (TensorElement a)
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

-- | Position of the minimum along one nonempty axis, removing that axis.
argmin
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , E.Ord (TensorElement a)
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
argmin = Raw.argmin @axis . toTensor

-- | Maximum of all elements of a statically nonempty tensor.
maximum
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , BackwardConstraints t (Fractional a, Fractional (TensorElement a))
       , E.Ord (TensorElement a)
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
       , E.Ord (TensorElement a)
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

-- | Position of the maximum along one nonempty axis, removing that axis.
argmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , E.Ord (TensorElement a)
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
argmax = Raw.argmax @axis . toTensor

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

-- | Logarithm of the sum of exponentials, using maximum subtraction for finite inputs.
logSumExp
    :: forall {t :: TensorLike} sh a
     . ( TensorOps t a
       , RealFloat a
       , Floating (TensorElement a)
       , E.Ord (TensorElement a)
       , E.RealFloat (TensorElement a)
       , Shape sh
       )
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
     . ( TensorOps t a
       , RealFloat a
       , Floating (TensorElement a)
       , E.Ord (TensorElement a)
       , E.RealFloat (TensorElement a)
       , Shape sh
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       )
    => t sh a
    -> t (ReduceShape xs sh) a
logSumExpAxes =
    withVJP
        (Proxy @(() :: Constraint))
        (Raw.logSumExpAxes @xs)
        (Pullback.logSumExpAxes @xs)

-- | Normalize exponentials along one axis without overflowing on large finite inputs.
softmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Floating a
       , Floating (TensorElement a)
       , E.Ord (TensorElement a)
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

-- | Logarithm of softmax along one nonempty axis, computed without taking the log of small probabilities.
logSoftmax
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Floating a
       , Floating (TensorElement a)
       , E.Ord (TensorElement a)
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
