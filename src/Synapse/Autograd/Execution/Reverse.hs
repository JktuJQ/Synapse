{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Reverse-mode execution with independent inputs on the <https://hackage.haskell.org/package/backprop backprop> tape.
module Synapse.Autograd.Execution.Reverse
  ( pullback,
  )
where

import Data.Functor.Identity (Identity (..))
import Data.Vinyl.Core (Rec (..))
import qualified Numeric.Backprop.Explicit as BPE
import Synapse.Autograd.Differentiable.Structure (Constructor, Field (..), Fields (..), GradientConstructor, Representation, Structure (..))
import Synapse.Autograd.Execution.Traversal (add, pack)
import Synapse.Autograd.Type (Context)
import qualified Synapse.Autograd.Type as AG

-- | Type-level concatenation of independently registered inputs.
type family xs :++ ys where
  '[] :++ ys = ys
  (x ': xs) :++ ys = x ': (xs :++ ys)

-- | Heterogeneous tape inputs with conversions to the public representation.
data Inputs a where
  Inputs ::
    Rec BPE.ZeroFunc xs ->
    Rec BPE.AddFunc xs ->
    Rec Identity xs ->
    (Rec Identity xs -> a) ->
    (forall s. Rec (BPE.BVar s) xs -> Representation s a) ->
    Inputs a

-- | Append input records without converting the stored values.
append :: Rec f xs -> Rec f ys -> Rec f (xs :++ ys)
append RNil ys = ys
append (x :& xs) ys = x :& append xs ys

-- | Split a record using the existing left input record as its structural witness.
split :: Rec g xs -> Rec f (xs :++ ys) -> (Rec f xs, Rec f ys)
split RNil values = (RNil, values)
split (_ :& rest) (value :& values) =
  let (left, right) = split rest values
   in (value :& left, right)

-- | Register each leaf as its own tape input.
inputs :: Structure a -> a -> Inputs a
inputs (Leaf wrap _) value =
  Inputs
    (BPE.ZF AG.zero :& RNil)
    (BPE.AF AG.add :& RNil)
    (Identity value :& RNil)
    (\(Identity x :& RNil) -> x)
    (\(x :& RNil) -> wrap (AG.Var x))
inputs (Record fields constructor differentiableConstructor) value = case inputFields fields value of
  FieldInputs zeros additions values rebuild variables ->
    Inputs
      zeros
      additions
      values
      (`rebuild` constructor)
      (\(arguments :: Rec (BPE.BVar s) leaves) -> variables arguments (differentiableConstructor @s))

-- | Independently registered field leaves with constructor application in declaration order.
data FieldInputs a fields where
  FieldInputs ::
    Rec BPE.ZeroFunc leaves ->
    Rec BPE.AddFunc leaves ->
    Rec Identity leaves ->
    (Rec Identity leaves -> Constructor fields a -> a) ->
    (forall s. Rec (BPE.BVar s) leaves -> GradientConstructor s fields a -> Representation s a) ->
    FieldInputs a fields

-- | Flatten nested field descriptions without registering the containing record as a single input.
inputFields :: Fields a fields -> a -> FieldInputs a fields
inputFields NoFields _ =
  FieldInputs
    RNil
    RNil
    RNil
    (\RNil value -> value)
    (\RNil value -> value)
inputFields (MoreFields (Field child getter _) rest) value =
  case (inputs child (getter value), inputFields rest value) of
    (Inputs leftZeros leftAdds leftValues rebuildLeft wrapLeft, FieldInputs rightZeros rightAdds rightValues rebuildRight wrapRight) ->
      FieldInputs
        (append leftZeros rightZeros)
        (append leftAdds rightAdds)
        (append leftValues rightValues)
        (\values constructor -> let (ls, rs) = split leftZeros values in rebuildRight rs (constructor (rebuildLeft ls)))
        (\(variables :: Rec (BPE.BVar s) leaves) constructor -> let (ls, rs) = split leftZeros variables in wrapRight @s rs (constructor (wrapLeft ls)))

-- | Run a reverse pass while registering each leaf as an independent input.
pullback ::
  Structure a ->
  Structure b ->
  (forall s. (Context s) => Representation s a -> Representation s b) ->
  a ->
  (b, b -> a)
pullback inputStructure resultStructure function input = case inputs inputStructure input of
  Inputs zeroFunctions additions values rebuild variables ->
    let (value, backward) =
          BPE.backpropWithN
            zeroFunctions
            (\(arguments :: Rec (BPE.BVar s) leaves) -> outputRoot resultStructure zeroFunctions additions arguments (pack @s resultStructure (function @s (variables arguments))))
            values
     in (value, rebuild . backward)

-- | Anchor the requested output to a live input so that it becomes the final tape node.
-- The extra edge contributes a reference-shaped zero directly to the input.
outputRoot :: (Context s) => Structure b -> Rec BPE.ZeroFunc xs -> Rec BPE.AddFunc xs -> Rec (BPE.BVar s) xs -> BPE.BVar s b -> BPE.BVar s b
outputRoot result zeros additions variables value = case (zeros, additions, variables) of
  (zero :& _, addition :& _, variable :& _) ->
    BPE.liftOp2
      (BPE.AF (add result))
      addition
      (BPE.op2 (\output reference -> (output, (,BPE.runZF zero reference))))
      value
      variable
  (RNil, RNil, RNil) -> value
