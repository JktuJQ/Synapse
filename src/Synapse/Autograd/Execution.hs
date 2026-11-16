{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

{- | Evaluation and reverse-mode differentiation of tensor functions.

The input and output may be individual tensors, tuples, or records described by 'TensorStructure'.
'WithGrad' replaces their tensors with differentiable tensors.
Its @s@ parameter identifies a differentiation scope supplied through 'Context'.
The differentiated function must work in any such scope; its variables cannot escape that scope.

'backprop' returns both the output and the input gradient, while 'gradient' returns only the gradient.
Both differentiate the sum of all elements in all output tensors.
Use 'pullback' to supply an explicit output seed instead.

Values and gradients are staged tensor expressions.
Execute them with the functions in "Synapse.Tensor.Backend" to obtain materialized tensors.
For example, with @DataKinds@ and @TypeApplications@ enabled:

@
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T

input = T.fromList \@'[2] [2, 3 :: Double]
(value, derivative) = AG.backprop (\\x -> x * x) input

-- [4, 9]
squares = T.toList $ T.run T.CPU value

-- [4, 6]
gradients = T.toList $ T.run T.CPU derivative
@

Use 'constant' for tensors that participate in the calculation but are not differentiation inputs.
Use 'stopGrad' to discard the differentiation history of a computed value.
-}
module Synapse.Autograd.Execution
    ( -- * Differentiable structures
      Differentiable

      -- * Evaluation and differentiation
    , evaluate
    , backprop
    , gradient
    , pullback

      -- * Gradient boundaries
    , constant
    , stopGrad
    ) where

import Data.Bifunctor (bimap)
import Data.Functor.Identity (Identity (..))
import Numeric.Backprop.Op (Rec (..))

import Synapse.Autograd.Record (append, split)
import Synapse.Autograd.Representation (Backpropagatable, Context)
import Synapse.Autograd.Tensor (WithGrad)
import Synapse.Tensor.Structure
    ( ProductFields
    , TensorConstraints
    , TensorProduct
    , TensorStructure
    , ValueTree (..)
    , flatten
    , rebuild
    , tensorConstraints
    )

import qualified Numeric.Backprop.Explicit as BPE
import qualified Numeric.Backprop.Internal as BPI

import qualified Synapse.Autograd.Representation as AG
import qualified Synapse.Autograd.Tensor.Representation as Tensor
import qualified Synapse.Tensor.Representation as T

{- | Tensor structures whose fields support gradient accumulation and default output seeds.
The instance is inferred from 'TensorStructure' and 'Backpropagatable' for every tensor field.
No separate instance is needed for a user-defined parameter record.
-}
class (TensorStructure a, TensorConstraints Backpropagatable a) => Differentiable a

instance (TensorStructure a, TensorConstraints Backpropagatable a) => Differentiable a

-- | Add corresponding tensor gradients.
add
    :: ValueTree Backpropagatable a
    -> a
    -> a
    -> a
add TensorValues = AG.add
add (PairValues left right) = \(x1, x2) (y1, y2) -> (add left x1 y1, add right x2 y2)

-- | Construct the default output seed for each tensor in a product.
ones
    :: ValueTree Backpropagatable a
    -> a
ones TensorValues = AG.one
ones (PairValues left right) = (ones left, ones right)

-- | Combine tensor variables into one variable, routing gradients back to the individual tensors.
pack
    :: forall s a
     . Context s
    => ValueTree Backpropagatable a
    -> ProductFields (Tensor.Tensor s) a
    -> BPE.BVar s a
pack TensorValues = Tensor.unTensor
pack (PairValues left right) = \(x, y) ->
    BPE.liftOp2
        (BPE.AF (add left))
        (BPE.AF (add right))
        (BPE.op2 (\a b -> ((a, b), id)))
        (pack @s left x)
        (pack @s right y)

-- | Wrap each tensor leaf as a constant variable, preserving the product structure.
constants
    :: forall s a
     . ValueTree Backpropagatable a
    -> a
    -> ProductFields (Tensor.Tensor s) a
constants TensorValues = Tensor.Tensor . BPE.auto
constants (PairValues left right) = bimap (constants @s left) (constants @s right)

-- | Stop gradient propagation through every tensor variable in a product.
stopped
    :: forall s a
     . ValueTree Backpropagatable a
    -> ProductFields (Tensor.Tensor s) a
    -> ProductFields (Tensor.Tensor s) a
stopped TensorValues = Tensor.Tensor . BPE.auto . BPI._bvVal . Tensor.unTensor
stopped (PairValues left right) = bimap (stopped @s left) (stopped @s right)

-- | Input values, gradient rules, and structure conversions for a differentiation run.
data ExecutionInputs a where
    ExecutionInputs
        :: Rec BPE.ZeroFunc xs
        -> Rec BPE.AddFunc xs
        -> Rec Identity xs
        -> (Rec Identity xs -> a)
        -> (forall s. Rec (BPE.BVar s) xs -> ProductFields (Tensor.Tensor s) a)
        -> ExecutionInputs a

-- | Prepare tensor values and conversions for a differentiation run.
prepareExecutionInputs
    :: ValueTree Backpropagatable a
    -> a
    -> ExecutionInputs a
prepareExecutionInputs TensorValues value =
    ExecutionInputs
        (BPE.ZF (const AG.zero) :& RNil)
        (BPE.AF AG.add :& RNil)
        (Identity value :& RNil)
        (\(Identity x :& RNil) -> x)
        (\(x :& RNil) -> Tensor.Tensor x)
prepareExecutionInputs (PairValues left right) (x, y) =
    case (prepareExecutionInputs left x, prepareExecutionInputs right y) of
        ( ExecutionInputs leftZeros leftAdds leftValues rebuildLeft wrapLeft
            , ExecutionInputs rightZeros rightAdds rightValues rebuildRight wrapRight
            ) ->
                ExecutionInputs
                    (append leftZeros rightZeros)
                    (append leftAdds rightAdds)
                    (append leftValues rightValues)
                    ( \arguments ->
                        let (ls, rs) = split leftZeros arguments
                         in (rebuildLeft ls, rebuildRight rs)
                    )
                    ( \(arguments :: Rec (BPE.BVar s) leaves) ->
                        let (ls, rs) = split leftZeros arguments
                         in (wrapLeft @s ls, wrapRight @s rs)
                    )

-- | Make the result the final tape node by attaching a zero contribution to a live input.
outputRoot
    :: Context s
    => ValueTree Backpropagatable b
    -> Rec BPE.ZeroFunc xs
    -> Rec BPE.AddFunc xs
    -> Rec (BPE.BVar s) xs
    -> BPE.BVar s b
    -> BPE.BVar s b
outputRoot result zeros additions variables value = case (zeros, additions, variables) of
    (zero :& _, addition :& _, variable :& _) ->
        BPE.liftOp2
            (BPE.AF (add result))
            addition
            (BPE.op2 (\output reference -> (output, (,BPE.runZF zero reference))))
            value
            variable
    (RNil, RNil, RNil) -> value

-- | Register independent tensor inputs and return the result with its pullback.
runReverse
    :: ValueTree Backpropagatable a
    -> ValueTree Backpropagatable b
    -> (forall s. Context s => ProductFields (Tensor.Tensor s) a -> ProductFields (Tensor.Tensor s) b)
    -> a
    -> (b, b -> a)
runReverse inputStructure resultStructure function input = case prepareExecutionInputs inputStructure input of
    ExecutionInputs zeroFunctions additions values restoreInputs variables ->
        let (value, backward) =
                BPE.backpropWithN
                    zeroFunctions
                    ( \(arguments :: Rec (BPE.BVar s) leaves) ->
                        outputRoot
                            resultStructure
                            zeroFunctions
                            additions
                            arguments
                            (pack @s resultStructure (function @s (variables arguments)))
                    )
                    values
         in (value, restoreInputs . backward)

-- | Evaluate a tensor function with constant inputs, without computing gradients.
evaluate
    :: forall a b
     . (Differentiable a, Differentiable b)
    => (forall s. Context s => WithGrad s a -> WithGrad s b)
    -> a
    -> b
evaluate f input = rebuild @_ @T.Tensor $ BPE.evalBP0 result
  where
    result
        :: forall s
         . Context s
        => BPE.BVar s (TensorProduct T.Tensor b)
    result =
        pack
            (tensorConstraints @b @Backpropagatable)
            (flatten @b @(Tensor.Tensor s) (f @s (constant @s @a input)))

{- | Return the output and the input gradient of the sum of all output tensor elements.
Unused inputs receive zero tensors; outputs wrapped with 'constant' or 'stopGrad' contribute no gradients.
-}
backprop
    :: forall a b
     . (Differentiable a, Differentiable b)
    => (forall s. Context s => WithGrad s a -> WithGrad s b)
    -> a
    -> (b, a)
backprop f input =
    let (output, backward) = pullback @a @b f input
        seed = rebuild @_ @T.Tensor (ones (tensorConstraints @b @Backpropagatable))
     in (output, backward seed)

-- | Return only the input gradient computed by 'backprop'.
gradient
    :: forall a b
     . (Differentiable a, Differentiable b)
    => (forall s. Context s => WithGrad s a -> WithGrad s b)
    -> a
    -> a
gradient f = snd . backprop @a @b f

{- | Return the output and its pullback, mapping an output seed to @J_f(input)^T * seed@.
Seeds must have the same structure and tensor shapes as the output; the result has the input structure.
-}
pullback
    :: forall a b
     . (Differentiable a, Differentiable b)
    => (forall s. Context s => WithGrad s a -> WithGrad s b)
    -> a
    -> (b, b -> a)
pullback f input =
    let wrapped
            :: forall s
             . Context s
            => ProductFields (Tensor.Tensor s) (TensorProduct T.Tensor a)
            -> ProductFields (Tensor.Tensor s) (TensorProduct T.Tensor b)
        wrapped = flatten @b @(Tensor.Tensor s) . f @s . rebuild @a @(Tensor.Tensor s)
        (output, backward) =
            runReverse
                (tensorConstraints @a @Backpropagatable)
                (tensorConstraints @b @Backpropagatable)
                wrapped
                (flatten @_ @T.Tensor input)
     in (rebuild @_ @T.Tensor output, rebuild @_ @T.Tensor . backward . flatten @_ @T.Tensor)

-- | Wrap tensors as constants that contribute no gradients to their source values.
constant
    :: forall s a
     . Differentiable a
    => a
    -> WithGrad s a
constant =
    rebuild @a @(Tensor.Tensor s)
        . constants @s (tensorConstraints @a @Backpropagatable)
        . flatten @_ @T.Tensor

{- | Preserve tensor values while stopping gradients through this use of them.
Other uses retain their histories: the gradient of @x * stopGrad x@ is @x@.
-}
stopGrad
    :: forall s a
     . Differentiable a
    => WithGrad s a
    -> WithGrad s a
stopGrad =
    rebuild @a @(Tensor.Tensor s)
        . stopped @s (tensorConstraints @a @Backpropagatable)
        . flatten @a @(Tensor.Tensor s)
