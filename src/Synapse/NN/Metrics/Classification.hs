{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Classification metrics with explicit label conventions.
module Synapse.NN.Metrics.Classification
    ( Averaging (..)
    , accuracy
    , precisionPerClass
    , precision
    , recallPerClass
    , recall
    , f1PerClass
    , f1
    , confusionMatrix
    , binaryAccuracy
    , binaryPrecision
    , binaryRecall
    , binaryF1
    ) where

import Synapse.Tensor.Ops.Definition (TensorOps (toTensor))
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..), TensorLike)
import Synapse.Tensor.Shapes.Axes (Axes, AxisSize, CanReduceNonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)

import qualified Data.Array.Accelerate as A

import qualified Synapse.Tensor.Ops.Construction as C
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Reduction as R

-- | Divide counts, returning zero when the denominator is zero.
ratio
    :: (Shape sh, DType a, Fractional a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => Tensor sh a
    -> Tensor sh a
    -> Tensor sh a
ratio numerator denominator = E.select (denominator E.== 0) 0 $ numerator / E.select (denominator E.== 0) 1 denominator

-- | Fraction of true elements; empty inputs return NaN.
meanMatches
    :: forall sh a
     . (Shape sh, DType a, RealFloat a, Fractional (TensorElement a))
    => Tensor sh Bool
    -> Tensor '[] a
meanMatches matches = R.mean (E.select matches (C.full @sh 1) (C.full @sh 0))

{- | Count true positives, predicted positives, and target support.

Invalid target labels are excluded from every count.
-}
classificationCounts
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Shape '[AxisSize axis sh]
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> (Tensor '[AxisSize axis sh] a, Tensor '[AxisSize axis sh] a, Tensor '[AxisSize axis sh] a)
classificationCounts predictions (Tensor targets) =
    let Tensor predicted = R.argmax @axis predictions
        classes = shapeSize @'[AxisSize axis sh]
        zeros = A.fill (A.index1 $ A.constant classes) $ unTensorElement (0 :: TensorElement a)
        ones = A.fill (A.shape targets) $ unTensorElement (1 :: TensorElement a)
        add
            :: A.Exp a
            -> A.Exp a
            -> A.Exp a
        add x y = unTensorElement $ TensorElement x + TensorElement y
        count correct usePrediction =
            Tensor $
                A.permute
                    add
                    zeros
                    ( \ix ->
                        let actual = targets A.! ix
                            guessed = predicted A.! ix
                            valid = actual A.>= 0 A.&& actual A.< A.constant classes
                            selected = if usePrediction then guessed else actual
                            include = if correct then valid A.&& guessed A.== actual else valid
                         in A.cond include (A.Just_ $ A.index1 selected) A.Nothing_
                    )
                    ones
     in (count True False, count False True, count False False)

-- | Compute a score from pooled counts or aggregate its class-wise values.
averageCounts
    :: forall classes a
     . (Shape '[classes], DType a, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => (forall sh. Shape sh => Tensor sh a -> Tensor sh a -> Tensor sh a -> Tensor sh a)
    -> Averaging
    -> (Tensor '[classes] a, Tensor '[classes] a, Tensor '[classes] a)
    -> Tensor '[] a
averageCounts score averaging (tp, predicted, support) =
    case averaging of
        Micro -> score (R.sum tp) (R.sum predicted) $ R.sum support
        Macro -> ratio (R.sum $ score tp predicted support) $ C.full @'[] $ fromIntegral $ shapeSize @'[classes]
        Weighted -> ratio (R.sum $ score tp predicted support * support) $ R.sum support

-- | Count positive-label true positives, predictions, and targets with an explicit threshold.
binaryCounts
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => a
    -> t sh a
    -> t sh a
    -> (Tensor '[] a, Tensor '[] a, Tensor '[] a)
binaryCounts threshold predictions targets =
    let predicted = toTensor predictions E.>= C.full @sh threshold
        actual = toTensor targets E.>= 0.5
        count mask = R.sum $ E.select mask (C.full @sh 1) $ C.full @sh 0
     in (count $ predicted E.&& actual, count predicted, count actual)

-- | Combine class-wise precision, recall, or F1 statistics.
data Averaging
    = Micro
    | Macro
    | Weighted
    deriving (Eq, Show)

{- | Fraction of correct multiclass predictions along the supplied class axis.

Predictions may be logits or probabilities; the greatest score determines the predicted class.
Targets contain zero-based integer class indices and have the prediction shape with the class axis removed.
Ties select the first class, as in argmax; empty observation sets return NaN.
-}
accuracy
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> Tensor '[] a
accuracy predictions targets = meanMatches @(ReduceShape '[axis] sh) @a (R.argmax @axis predictions E.== targets)

{- | Precision @TP / (TP + FP)@ for each class in class-axis order.

Predictions are logits or probabilities; targets are zero-based class indices with the class axis removed.
All other dimensions are pooled, and argmax ties select the first class.
Out-of-range targets are ignored; a zero denominator gives zero.
Absent classes and empty observation sets therefore have a score of zero.
-}
precisionPerClass
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Shape '[AxisSize axis sh]
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> Tensor '[AxisSize axis sh] a
precisionPerClass predictions targets =
    let (tp, predicted, _) = classificationCounts @axis predictions targets
     in ratio tp predicted

{- | Precision @TP / (TP + FP)@, using the supplied class averaging strategy.

Label conventions and zero-denominator behavior match 'precisionPerClass'.
Micro pools counts, macro averages scores over all declared classes, and weighted uses target support.
For epoch-level scores, pool observations or counts rather than averaging batch scores.
-}
precision
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Shape '[AxisSize axis sh]
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => Averaging
    -> t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> Tensor '[] a
precision averaging predictions targets =
    averageCounts (\tp predicted _ -> ratio tp predicted) averaging $
        classificationCounts @axis predictions targets

{- | Recall @TP / (TP + FN)@ for each class in class-axis order.

Predictions are logits or probabilities; targets are zero-based class indices with the class axis removed.
All other dimensions are pooled, and argmax ties select the first class.
Out-of-range targets are ignored; a zero denominator gives zero.
Absent classes and empty observation sets therefore have a score of zero.
-}
recallPerClass
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Shape '[AxisSize axis sh]
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> Tensor '[AxisSize axis sh] a
recallPerClass predictions targets =
    let (tp, _, support) = classificationCounts @axis predictions targets
     in ratio tp support

{- | Recall @TP / (TP + FN)@, using the supplied class averaging strategy.

Label conventions and zero-denominator behavior match 'recallPerClass'.
Micro pools counts, macro averages scores over all declared classes, and weighted uses target support.
For epoch-level scores, pool observations or counts rather than averaging batch scores.
-}
recall
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Shape '[AxisSize axis sh]
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => Averaging
    -> t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> Tensor '[] a
recall averaging predictions targets =
    averageCounts (\tp _ support -> ratio tp support) averaging $ classificationCounts @axis predictions targets

{- | F1 score @2 * TP / (2 * TP + FP + FN)@ for each class in class-axis order.

Predictions are logits or probabilities; targets are zero-based class indices with the class axis removed.
All other dimensions are pooled, and argmax ties select the first class.
Out-of-range targets are ignored; a zero denominator gives zero.
Absent classes and empty observation sets therefore have a score of zero.
-}
f1PerClass
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Shape '[AxisSize axis sh]
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> Tensor '[AxisSize axis sh] a
f1PerClass predictions targets =
    let (tp, predicted, support) = classificationCounts @axis predictions targets
     in ratio (2 * tp) $ predicted + support

{- | F1 score @2 * TP / (2 * TP + FP + FN)@, using the supplied class averaging strategy.

Label conventions and zero-denominator behavior match 'f1PerClass'.
Micro pools counts, macro averages scores over all declared classes, and weighted uses target support.
For epoch-level scores, pool observations or counts rather than averaging batch scores.
-}
f1
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Shape '[AxisSize axis sh]
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , RealFloat a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => Averaging
    -> t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> Tensor '[] a
f1 averaging predictions targets =
    averageCounts (\tp predicted support -> ratio (2 * tp) $ predicted + support) averaging $
        classificationCounts @axis predictions targets

{- | Integer confusion matrix with target classes on rows and predicted classes on columns.

Predictions may be logits or probabilities; ties choose the first class.
Observation dimensions are pooled, out-of-range targets are ignored, and empty observations give a zero matrix.
Unlike precision, recall, and F1, this function allocates a class-by-class matrix.
-}
confusionMatrix
    :: forall {t :: TensorLike} axis sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , Shape '[AxisSize axis sh]
       , Axes '[axis]
       , CanReduceNonEmpty '[axis] sh
       , E.Ord (TensorElement a)
       )
    => t sh a
    -> Tensor (ReduceShape '[axis] sh) Int
    -> Tensor '[AxisSize axis sh, AxisSize axis sh] Int
confusionMatrix predictions (Tensor targets) =
    let Tensor predicted = R.argmax @axis predictions
        classes = shapeSize @'[AxisSize axis sh]
        zeros = A.fill (A.index2 (A.constant classes) (A.constant classes)) 0
        ones = A.fill (A.shape targets) 1
        destination ix =
            let actual = targets A.! ix
             in A.cond
                    (actual A.>= 0 A.&& actual A.< A.constant classes)
                    (A.Just_ $ A.index2 actual $ predicted A.! ix)
                    A.Nothing_
     in Tensor $ A.permute (+) zeros destination ones

{- | Fraction of correct binary or independent multi-label predictions.

Use threshold 0 for logits or 0.5 for probabilities; a prediction equal to the threshold is positive.
Targets at least 0.5 are positive, including ordinary zero/one labels.
All elements count independently; empty inputs return NaN.
-}
binaryAccuracy
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => a
    -> t sh a
    -> t sh a
    -> Tensor '[] a
binaryAccuracy threshold predictions targets =
    let predicted = toTensor predictions E.>= C.full @sh threshold
        actual = toTensor targets E.>= 0.5
     in meanMatches @sh @a (predicted E.== actual)

{- | Precision @TP / (TP + FP)@ for the positive binary label.

Use threshold 0 for logits or 0.5 for probabilities; values equal to the threshold are positive.
Targets at least 0.5 are positive, and all elements are pooled for independent multi-label predictions.
Zero denominators and empty inputs give zero.
-}
binaryPrecision
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => a
    -> t sh a
    -> t sh a
    -> Tensor '[] a
binaryPrecision threshold predictions targets =
    let (tp, predicted, _) = binaryCounts threshold predictions targets
     in ratio tp predicted

{- | Recall @TP / (TP + FN)@ for the positive binary label.

Use threshold 0 for logits or 0.5 for probabilities; values equal to the threshold are positive.
Targets at least 0.5 are positive, and all elements are pooled for independent multi-label predictions.
Zero denominators and empty inputs give zero.
-}
binaryRecall
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => a
    -> t sh a
    -> t sh a
    -> Tensor '[] a
binaryRecall threshold predictions targets =
    let (tp, _, support) = binaryCounts threshold predictions targets
     in ratio tp support

{- | F1 score @2 * TP / (2 * TP + FP + FN)@ for the positive binary label.

Use threshold 0 for logits or 0.5 for probabilities; values equal to the threshold are positive.
Targets at least 0.5 are positive, and all elements are pooled for independent multi-label predictions.
Zero denominators and empty inputs give zero.
-}
binaryF1
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh, RealFloat a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => a
    -> t sh a
    -> t sh a
    -> Tensor '[] a
binaryF1 threshold predictions targets =
    let (tp, predicted, support) = binaryCounts threshold predictions targets
     in ratio (2 * tp) $ predicted + support
