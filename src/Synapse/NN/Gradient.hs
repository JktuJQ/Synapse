{-# LANGUAGE TypeFamilies #-}

{- | Global gradient norms and clipping for structures with one element type.
Every tensor field contributes to one norm; clipping applies the same factor to all fields.
Norms, limits, and scaling use the gradients' element type.
-}
module Synapse.NN.Gradient
    ( gradientNorm
    , gradientNormL1
    , gradientNormL2
    , clipGradNorm
    , clipGradNormL1
    , clipGradNormL2
    ) where

import Synapse.Tensor.Ops.Constraints (FractionalDType, NumDType, OrdDType, RealFloatDType)
import Synapse.Tensor.Representation (Tensor)
import Synapse.Tensor.Structure (ElementConstraints, TensorStructure)

import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Structure as Structure

{- | Reduce each tensor to a partial norm and combine partial norms from left to right.
The accumulator starts at zero; the combining function must treat zero as its identity.
-}
gradientNorm
    :: forall a p
     . (TensorStructure p, ElementConstraints ((~) a) p, NumDType a)
    => (forall sh. T.Shape sh => Tensor sh a -> Tensor '[] a)
    -> (Tensor '[] a -> Tensor '[] a -> Tensor '[] a)
    -> p
    -> Tensor '[] a
gradientNorm norm combine =
    Structure.foldElements @((~) a) @p @Tensor (\_ total tensor -> combine total $ norm tensor) 0

-- | Sum absolute gradient elements across every tensor field.
gradientNormL1
    :: forall a p
     . (TensorStructure p, ElementConstraints ((~) a) p, NumDType a)
    => p
    -> Tensor '[] a
gradientNormL1 = gradientNorm T.normL1 (+)

-- | L2 norm across every tensor field, using one shared floating-point element type.
gradientNormL2
    :: forall a p
     . (TensorStructure p, ElementConstraints ((~) a) p, RealFloatDType a)
    => p
    -> Tensor '[] a
gradientNormL2 =
    gradientNorm T.normL2 (\left right -> T.normL2 $ T.stack @0 left right)

-- | Scale every field by the same factor when the norm exceeds the limit.
clipGradNorm
    :: forall a p
     . (TensorStructure p, ElementConstraints ((~) a) p, FractionalDType a, OrdDType a)
    => (p -> Tensor '[] a)
    -> a
    -> p
    -> p
clipGradNorm norm maxNorm gradients =
    let magnitude = norm gradients
        limit = T.full maxNorm
        safeNorm = T.select (magnitude T.== 0) 1 magnitude
        factor = T.select (magnitude T.<= limit) 1 (limit / safeNorm)
     in Structure.mapElements @((~) a) @p @Tensor @Tensor
            (\_ tensor -> T.broadcast factor * tensor)
            gradients

{- | Bound the global L1 norm by scaling all fields together in their shared element type.

Supply a finite, nonnegative limit. Zero gradients are preserved; nonfinite gradients are not repaired.
-}
clipGradNormL1
    :: (TensorStructure p, ElementConstraints ((~) a) p, FractionalDType a, OrdDType a)
    => a
    -> p
    -> p
clipGradNormL1 = clipGradNorm gradientNormL1

{- | Bound the global L2 norm by scaling all fields together.

Supply a finite, nonnegative limit. Zero gradients are preserved; nonfinite gradients are not repaired.
Scaling uses the same precision as the gradients, so very small factors may underflow.
Apply this after differentiation and before an optimizer step.

@
let gradients = AG.gradient objective parameters
    clipped = NN.clipGradNormL2 1 gradients
    (updated, nextState) = NN.runState
      (NN.optimizerStep optimizer parameters clipped)
      optimizerState
@
-}
clipGradNormL2
    :: (TensorStructure p, ElementConstraints ((~) a) p, RealFloatDType a)
    => a
    -> p
    -> p
clipGradNormL2 = clipGradNorm gradientNormL2
