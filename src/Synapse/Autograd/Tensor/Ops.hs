{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Register tensor primitives on the differentiation tape.
module Synapse.Autograd.Tensor.Ops () where

import qualified Data.Array.Accelerate as A
import Data.Functor.Identity (Identity (..))
import qualified Numeric.Backprop.Explicit as BPE
import Numeric.Backprop.Op (Rec (..))
import Synapse.Autograd.Tensor.Type (Tensor (..), TensorElement (..))
import Synapse.Autograd.Variable (Context, Var (..), add, liftUnary, variableValue)
import Synapse.Tensor.Ops (TensorOps (..))
import Synapse.Tensor.Type (DType)
import qualified Synapse.Tensor.Type as T

-- | Register a primitive with an accumulation function for each independently shaped input.
liftTensor :: (Context s) => Rec BPE.AddFunc inputs -> (Rec Identity inputs -> (T.Tensor out b, T.Tensor out b -> Rec Identity inputs)) -> Rec (BPE.BVar s) inputs -> Tensor s out b
liftTensor additions rule = Tensor . Var . BPE.liftOp additions (BPE.Op rule)

instance (Context s, DType a, Fractional a, Fractional (T.TensorElement a)) => TensorOps (Tensor s) a where
  literal = Tensor . Var . BPE.auto
  primal = variableValue . unTensor
  fromElement (TensorElement x) = Tensor $ liftUnary add (\(T.TensorElement value) -> (T.Tensor (A.unit value), \(T.Tensor seed) -> T.TensorElement (A.the seed))) x
  liftElement forward backward (Tensor x) = TensorElement $ liftUnary add rule x
    where
      rule input =
        let output = forward input
         in (output, backward input output)
  lift1 forward backward (Tensor (Var x1)) =
    liftTensor
      (BPE.AF (+) :& RNil)
      ( \(Identity v1 :& RNil) ->
          let output = forward v1
           in ( output,
                \gradient ->
                  let g1 = backward v1 output gradient
                   in Identity g1 :& RNil
              )
      )
      (x1 :& RNil)

  lift2 forward backward (Tensor (Var x1)) (Tensor (Var x2)) =
    liftTensor
      (BPE.AF (+) :& BPE.AF (+) :& RNil)
      ( \(Identity v1 :& Identity v2 :& RNil) ->
          let output = forward v1 v2
           in ( output,
                \gradient ->
                  let (g1, g2) = backward v1 v2 output gradient
                   in Identity g1 :& Identity g2 :& RNil
              )
      )
      (x1 :& x2 :& RNil)

  lift3 forward backward (Tensor (Var x1)) (Tensor (Var x2)) (Tensor (Var x3)) =
    liftTensor
      (BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& RNil)
      ( \(Identity v1 :& Identity v2 :& Identity v3 :& RNil) ->
          let output = forward v1 v2 v3
           in ( output,
                \gradient ->
                  let (g1, g2, g3) = backward v1 v2 v3 output gradient
                   in Identity g1 :& Identity g2 :& Identity g3 :& RNil
              )
      )
      (x1 :& x2 :& x3 :& RNil)

  lift4 forward backward (Tensor (Var x1)) (Tensor (Var x2)) (Tensor (Var x3)) (Tensor (Var x4)) =
    liftTensor
      (BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& RNil)
      ( \(Identity v1 :& Identity v2 :& Identity v3 :& Identity v4 :& RNil) ->
          let output = forward v1 v2 v3 v4
           in ( output,
                \gradient ->
                  let (g1, g2, g3, g4) = backward v1 v2 v3 v4 output gradient
                   in Identity g1 :& Identity g2 :& Identity g3 :& Identity g4 :& RNil
              )
      )
      (x1 :& x2 :& x3 :& x4 :& RNil)

  lift5 forward backward (Tensor (Var x1)) (Tensor (Var x2)) (Tensor (Var x3)) (Tensor (Var x4)) (Tensor (Var x5)) =
    liftTensor
      (BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& RNil)
      ( \(Identity v1 :& Identity v2 :& Identity v3 :& Identity v4 :& Identity v5 :& RNil) ->
          let output = forward v1 v2 v3 v4 v5
           in ( output,
                \gradient ->
                  let (g1, g2, g3, g4, g5) = backward v1 v2 v3 v4 v5 output gradient
                   in Identity g1 :& Identity g2 :& Identity g3 :& Identity g4 :& Identity g5 :& RNil
              )
      )
      (x1 :& x2 :& x3 :& x4 :& x5 :& RNil)

  lift6 forward backward (Tensor (Var x1)) (Tensor (Var x2)) (Tensor (Var x3)) (Tensor (Var x4)) (Tensor (Var x5)) (Tensor (Var x6)) =
    liftTensor
      (BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& RNil)
      ( \(Identity v1 :& Identity v2 :& Identity v3 :& Identity v4 :& Identity v5 :& Identity v6 :& RNil) ->
          let output = forward v1 v2 v3 v4 v5 v6
           in ( output,
                \gradient ->
                  let (g1, g2, g3, g4, g5, g6) = backward v1 v2 v3 v4 v5 v6 output gradient
                   in Identity g1 :& Identity g2 :& Identity g3 :& Identity g4 :& Identity g5 :& Identity g6 :& RNil
              )
      )
      (x1 :& x2 :& x3 :& x4 :& x5 :& x6 :& RNil)

  lift7 forward backward (Tensor (Var x1)) (Tensor (Var x2)) (Tensor (Var x3)) (Tensor (Var x4)) (Tensor (Var x5)) (Tensor (Var x6)) (Tensor (Var x7)) =
    liftTensor
      (BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& RNil)
      ( \(Identity v1 :& Identity v2 :& Identity v3 :& Identity v4 :& Identity v5 :& Identity v6 :& Identity v7 :& RNil) ->
          let output = forward v1 v2 v3 v4 v5 v6 v7
           in ( output,
                \gradient ->
                  let (g1, g2, g3, g4, g5, g6, g7) = backward v1 v2 v3 v4 v5 v6 v7 output gradient
                   in Identity g1 :& Identity g2 :& Identity g3 :& Identity g4 :& Identity g5 :& Identity g6 :& Identity g7 :& RNil
              )
      )
      (x1 :& x2 :& x3 :& x4 :& x5 :& x6 :& x7 :& RNil)

  lift8 forward backward (Tensor (Var x1)) (Tensor (Var x2)) (Tensor (Var x3)) (Tensor (Var x4)) (Tensor (Var x5)) (Tensor (Var x6)) (Tensor (Var x7)) (Tensor (Var x8)) =
    liftTensor
      (BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& RNil)
      ( \(Identity v1 :& Identity v2 :& Identity v3 :& Identity v4 :& Identity v5 :& Identity v6 :& Identity v7 :& Identity v8 :& RNil) ->
          let output = forward v1 v2 v3 v4 v5 v6 v7 v8
           in ( output,
                \gradient ->
                  let (g1, g2, g3, g4, g5, g6, g7, g8) = backward v1 v2 v3 v4 v5 v6 v7 v8 output gradient
                   in Identity g1 :& Identity g2 :& Identity g3 :& Identity g4 :& Identity g5 :& Identity g6 :& Identity g7 :& Identity g8 :& RNil
              )
      )
      (x1 :& x2 :& x3 :& x4 :& x5 :& x6 :& x7 :& x8 :& RNil)

  lift9 forward backward (Tensor (Var x1)) (Tensor (Var x2)) (Tensor (Var x3)) (Tensor (Var x4)) (Tensor (Var x5)) (Tensor (Var x6)) (Tensor (Var x7)) (Tensor (Var x8)) (Tensor (Var x9)) =
    liftTensor
      (BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& BPE.AF (+) :& RNil)
      ( \(Identity v1 :& Identity v2 :& Identity v3 :& Identity v4 :& Identity v5 :& Identity v6 :& Identity v7 :& Identity v8 :& Identity v9 :& RNil) ->
          let output = forward v1 v2 v3 v4 v5 v6 v7 v8 v9
           in ( output,
                \gradient ->
                  let (g1, g2, g3, g4, g5, g6, g7, g8, g9) = backward v1 v2 v3 v4 v5 v6 v7 v8 v9 output gradient
                   in Identity g1 :& Identity g2 :& Identity g3 :& Identity g4 :& Identity g5 :& Identity g6 :& Identity g7 :& Identity g8 :& Identity g9 :& RNil
              )
      )
      (x1 :& x2 :& x3 :& x4 :& x5 :& x6 :& x7 :& x8 :& x9 :& RNil)
