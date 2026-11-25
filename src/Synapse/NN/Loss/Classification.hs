{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

-- | Classification loss functions with caller-supplied reductions.
module Synapse.NN.Loss.Classification
    ( binaryCrossEntropy
    , binaryCrossEntropyWithLogits
    , negativeLogLikelihood
    , crossEntropy
    , unsafeNegativeLogLikelihood
    , unsafeCrossEntropy
    , smoothLabels
    , unsafeCrossEntropySmoothed
    ) where

import Data.Array.Accelerate qualified as A
import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.Ops.Constraints (FloatingDType, NumDType, OrdDType)
import Synapse.Tensor.Ops.Definition (TensorOps, withCachedVJP, withVJP)
import Synapse.Tensor.Representation (Tensor (..), TensorElement (..), TensorLike)
import Synapse.Tensor.Shapes.Axes (Axes, CanReduceNonEmpty, ReduceShape, RemoveAxis)
import Synapse.Tensor.Shapes.Backend (backendCoordinates, backendIndex)
import Synapse.Tensor.Shapes.Dimensions (Shape, shape)

import Synapse.Autograd.Tensor.Ops.Pullback.Index qualified as Pullback
import Synapse.Tensor.Ops.Logic qualified as E
import Synapse.Tensor.Ops.Raw.Elementwise qualified as Elementwise
import Synapse.Tensor.Ops.Raw.Index qualified as Raw
import Synapse.Tensor.Ops.Raw.Reduction qualified as RawReduction
import Synapse.Tensor.Ops.Reduction qualified as R

-- | Reflect the size of an axis whose validity is established by the calling loss.
classCount
    :: forall axis sh
     . (KnownNat axis, Shape sh)
    => Int
classCount = shape @sh !! fromInteger (toInteger $ natVal $ Proxy @axis)

{- | Binary cross entropy @-(y * log p + (1 - y) * log (1 - p))@ from probabilities.

Predictions and targets have the same shape and must lie in @[0, 1]@.
Logarithms are bounded below by -100 so endpoint probabilities have finite losses.
Prefer 'binaryCrossEntropyWithLogits' when the model returns unnormalized scores.
-}
binaryCrossEntropy
    :: forall {t :: TensorLike} sh out_sh a
     . (TensorOps t a, Shape sh, RealFloat a, FloatingDType a, OrdDType a)
    => (t sh a -> t out_sh a)
    -> t sh a
    -> t sh a
    -> t out_sh a
binaryCrossEntropy reduce probabilities targets =
    reduce $
        ( curry $
            withVJP
                (Proxy @(() :: Constraint))
                (uncurry forward)
                (\(left, right) output -> backward left right output)
        )
            probabilities
            targets
  where
    forward
        :: Tensor sh a
        -> Tensor sh a
        -> Tensor sh a
    forward p y = negate (y * E.max (log p) (-100) + (1 - y) * E.max (log (1 - p)) (-100))
    backward
        :: Tensor sh a
        -> Tensor sh a
        -> Tensor sh a
        -> Tensor sh a
        -> (Tensor sh a, Tensor sh a)
    backward p y _ seed =
        ( seed * (p - y) / E.max (p * (1 - p)) 1e-12
        , seed * (log (1 - p) - log p)
        )

{- | Binary cross entropy from logits, equivalent to applying sigmoid followed by 'binaryCrossEntropy' away from its numerical limits.

Computes @max x 0 - x * y + log (1 + exp (-abs x))@ without overflowing exponentials for finite logits.
Targets must have the same shape and contain probabilities in @[0, 1]@.
Suitable for both binary classification and independent multi-label predictions.
Keep the model's final output unnormalized and apply sigmoid separately when probabilities are needed.
-}
binaryCrossEntropyWithLogits
    :: forall {t :: TensorLike} sh out_sh a
     . (TensorOps t a, Shape sh, FloatingDType a, OrdDType a)
    => (t sh a -> t out_sh a)
    -> t sh a
    -> t sh a
    -> t out_sh a
binaryCrossEntropyWithLogits reduce logits targets =
    reduce $ withVJP (Proxy @(() :: Constraint)) forward backward (logits, targets)
  where
    forward (x, y) = Elementwise.zipWith loss x y
    loss
        :: TensorElement a
        -> TensorElement a
        -> TensorElement a
    loss x y =
        let z = exp (negate (abs x))
         in E.max x 0 - x * y + 2 * atanh (z / (2 + z))
    backward (x, y) _ seed = (Elementwise.zipWith3 logitGradient x y seed, negate (seed * x))
    logitGradient
        :: TensorElement a
        -> TensorElement a
        -> TensorElement a
        -> TensorElement a
    logitGradient x y seed =
        let z = exp (negate (abs x))
            tailProbability = z / (1 + z)
         in seed * E.select (x E.>= 0) ((1 - y) - tailProbability) (tailProbability - y)

{- | Negative log likelihood @-sum (targets * logProbabilities)@ along the supplied class axis, followed by the caller's reduction.

Inputs have the same shape; targets are one-hot labels or class probabilities summing to one along the class axis.
The class axis must be nonempty and is removed before applying the reduction.
For probabilities p, pass @log p@; prefer 'crossEntropy' on logits to avoid underflow before taking logarithms.
-}
negativeLogLikelihood
    :: forall {t :: TensorLike} axis sh out_sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , NumDType a
       , Num (t sh a)
       )
    => (t (ReduceShape '[axis] sh) a -> t out_sh a)
    -> t sh a
    -> t sh a
    -> t out_sh a
negativeLogLikelihood reduce logProbabilities targets =
    reduce $ R.sumAxes @'[axis] $ negate $ targets * logProbabilities

{- | Multiclass cross entropy from unnormalized logits and one-hot or probability targets.

Computes @-sum (targets * logSoftmax logits)@ along the supplied class axis, then applies the reduction.
Targets have the same shape as logits and must form probability distributions along the class axis.
No sigmoid or softmax should be applied to model outputs before this loss.
The class axis must be nonempty; all remaining axes describe independent observations.
For shape @'[batch, classes]@ use @crossEntropy \@1 T.mean logits targets@.
-}
crossEntropy
    :: forall {t :: TensorLike} axis sh out_sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , FloatingDType a
       , OrdDType a
       , Num (t sh a)
       )
    => (t (ReduceShape '[axis] sh) a -> t out_sh a)
    -> t sh a
    -> t sh a
    -> t out_sh a
crossEntropy reduce logits = negativeLogLikelihood @axis reduce (R.logSoftmax @axis logits)

{- | Negative log likelihood from log probabilities and unchecked integer class labels.

Targets have the input shape with the class axis removed and are not differentiated.
Each label must lie in @[0, classes)@; invalid labels may cause invalid memory access.
No one-hot target tensor is constructed.
-}
unsafeNegativeLogLikelihood
    :: forall {t :: TensorLike} axis sh out_sh a
     . ( TensorOps t a
       , KnownNat axis
       , Shape sh
       , Shape (RemoveAxis axis sh)
       , CanReduceNonEmpty '[axis] sh
       , Num (t (RemoveAxis axis sh) a)
       )
    => (t (RemoveAxis axis sh) a -> t out_sh a)
    -> t sh a
    -> Tensor (RemoveAxis axis sh) Int
    -> t out_sh a
unsafeNegativeLogLikelihood reduce logProbabilities targets =
    reduce
        ( negate
            ( withVJP
                (Proxy @(Num (TensorElement a)))
                (\input -> Raw.unsafeGatherReduced @axis input targets)
                (Pullback.gatherReduced @axis targets)
                logProbabilities
            )
        )

{- | Multiclass cross entropy from logits and unchecked integer class labels.

For logits @'[batch, classes]@, targets have shape @'[batch]@ and the class axis is @1@.
Each label must lie in @[0, classes)@; invalid labels may cause invalid memory access.
Targets carry no differentiation history; the loss propagates gradients through logits.
Use 'crossEntropy' for one-hot labels or probability distributions instead.
-}
unsafeCrossEntropy
    :: forall {t :: TensorLike} axis sh out_sh a
     . ( TensorOps t a
       , KnownNat axis
       , Shape sh
       , Shape (RemoveAxis axis sh)
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , FloatingDType a
       , OrdDType a
       , Num (t (RemoveAxis axis sh) a)
       )
    => (t (RemoveAxis axis sh) a -> t out_sh a)
    -> t sh a
    -> Tensor (RemoveAxis axis sh) Int
    -> t out_sh a
unsafeCrossEntropy reduce logits targets =
    reduce $ withCachedVJP (Proxy @(() :: Constraint)) forward backward logits
  where
    forward input =
        let logProbabilities = RawReduction.logSoftmax @axis input
         in (negate (Raw.unsafeGatherReduced @axis logProbabilities targets), logProbabilities)
    backward (Tensor logProbabilities) (Tensor seeds) =
        Tensor $ A.generate (A.shape logProbabilities) $ \ix ->
            let coords = backendCoordinates @sh ix
                observation = backendIndex @(RemoveAxis axis sh) (take classAxis coords ++ drop (classAxis + 1) coords)
                Tensor labels = targets
                selected = coords !! classAxis A.== labels A.! observation
                logProbability = TensorElement $ logProbabilities A.! ix
                indicator = TensorElement $ A.cond selected (unTensorElement (1 :: TensorElement a)) (unTensorElement 0)
                seed = TensorElement $ seeds A.! observation
             in unTensorElement $ seed * (exp logProbability - indicator)
    classAxis = fromIntegral (natVal (Proxy @axis))

{- | Mix class probabilities with a uniform distribution: @(1 - smoothing) * targets + smoothing / classes@.

Smoothing must lie in @[0, 1]@ and the class axis must be nonempty.
Targets retain their shape and differentiation history.
-}
smoothLabels
    :: forall {t :: TensorLike} axis sh a
     . (KnownNat axis, Shape sh, CanReduceNonEmpty '[axis] sh, RealFloat a, Fractional (t sh a))
    => a
    -> t sh a
    -> t sh a
smoothLabels smoothing targets =
    let classes = classCount @axis @sh
     in realToFrac (1 - smoothing) * targets + realToFrac (smoothing / fromIntegral classes)

{- | Cross entropy with label smoothing and unchecked integer class labels.

Smoothing mixes the selected class with a uniform distribution without constructing one-hot targets.
A smoothing value of zero gives 'unsafeCrossEntropy', and one uses only the uniform distribution.
Smoothing must lie in @[0, 1]@; every target must lie in @[0, classes)@.
Invalid targets may cause invalid memory access; use 'smoothLabels' and 'crossEntropy' for probability targets.
-}
unsafeCrossEntropySmoothed
    :: forall {t :: TensorLike} axis sh out_sh a
     . ( TensorOps t a
       , KnownNat axis
       , Shape sh
       , Shape (RemoveAxis axis sh)
       , RemoveAxis axis sh ~ ReduceShape '[axis] sh
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , FloatingDType a
       , OrdDType a
       , Fractional (t (RemoveAxis axis sh) a)
       )
    => (t (RemoveAxis axis sh) a -> t out_sh a)
    -> a
    -> t sh a
    -> Tensor (RemoveAxis axis sh) Int
    -> t out_sh a
unsafeCrossEntropySmoothed reduce smoothing logits targets
    | smoothing == 0 = reduce selected
    | smoothing == 1 = reduce uniform
    | otherwise = reduce $ realToFrac (1 - smoothing) * selected + realToFrac smoothing * uniform
  where
    logProbabilities = R.logSoftmax @axis logits
    selected = unsafeNegativeLogLikelihood @axis id logProbabilities targets
    uniform = negate (R.sumAxes @'[axis] logProbabilities) / fromIntegral (classCount @axis @sh)
