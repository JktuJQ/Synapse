{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Backend loops and private state histories for fixed-length sequence scans.
module Synapse.Tensor.Ops.Primitives.Sequence where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat, natVal)
import Synapse.Tensor.Backend.Structure (BackendTree (..))
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import Synapse.Tensor.Ops.Primitives.Construction (full)
import Synapse.Tensor.Ops.Primitives.Math ()
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM, backendShape, shapeSize)
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..))

-- | Numeric array products with one history buffer per tensor leaf.
class (BackendTree value, A.Arrays (History value)) => ScanValues a value where
  type History value
  zeroValues :: value
  addValues :: value -> value -> value
  emptyHistory :: Int -> A.Acc (History value)
  saveHistory :: A.Exp Int -> value -> A.Acc (History value) -> A.Acc (History value)
  readHistory :: A.Exp Int -> A.Acc (History value) -> value

instance (Shape sh, DType a, Num a, Num (TensorElement a)) => ScanValues a (Tensor sh a) where
  type History (Tensor sh a) = A.Matrix a
  zeroValues = 0
  addValues = (+)
  emptyHistory count = A.fill (A.constant $ A.Z A.:. count A.:. shapeSize @sh) (A.constant 0)
  saveHistory i (Tensor value) history =
    A.permute const history (A.Just_ . A.index2 i . A.indexHead) $ A.flatten value
  readHistory i history = Tensor $
    A.reshape (A.constant $ backendShape $ Proxy @sh) $
      A.generate (A.constant $ A.Z A.:. shapeSize @sh) $
        \ix -> history A.! A.index2 i (A.indexHead ix)

instance (ScanValues a left, ScanValues a right) => ScanValues a (left, right) where
  type History (left, right) = (History left, History right)
  zeroValues = (zeroValues @a, zeroValues @a)
  addValues (x, y) (u, v) = (addValues @a x u, addValues @a y v)
  emptyHistory count = A.lift (emptyHistory @a @left count, emptyHistory @a @right count)
  saveHistory i (x, y) history =
    let (left, right) = A.unlift history
     in A.lift (saveHistory @a i x left, saveHistory @a i y right)
  readHistory i history =
    let (left, right) = A.unlift history
     in (readHistory @a i left, readHistory @a i right)

-- | Extract one leading-axis item without introducing a scalar differentiation input.
item :: forall sh sequence a. (Shape sh, Shape sequence, DType a) => A.Exp Int -> Tensor sequence a -> Tensor sh a
item i (Tensor input) = Tensor $
  A.reshape (A.constant $ backendShape $ Proxy @sh) $
    A.generate (A.constant $ A.Z A.:. shapeSize @sh) $ \ix ->
      A.flatten input A.! A.index1 (i * A.constant (shapeSize @sh) + A.indexHead ix)

-- | Write one leading-axis item into its disjoint output range.
writeItem :: forall sh sequence a. (Shape sh, Shape sequence, DType a) => A.Exp Int -> Tensor sh a -> Tensor sequence a -> Tensor sequence a
writeItem i (Tensor value) (Tensor target) =
  Tensor $
    A.reshape (A.shape target) $
      A.permute const (A.flatten target) (\ix -> A.Just_ $ A.index1 $ i * A.constant (shapeSize @sh) + A.indexHead ix) $
        A.flatten value

-- | Run without recording state histories. The loop graph is independent of the sequence length.
forward ::
  forall n p state input output a.
  ( KnownNat n,
    DType a,
    Num a,
    Num (TensorElement a),
    Shape input,
    Shape output,
    Shape (n ': input),
    Shape (n ': output),
    Flatten Tensor state,
    BackendTree (Leaves Tensor state)
  ) =>
  (p -> Tensor input a -> state -> (Tensor output a, state)) ->
  (p, (Tensor (n ': input) a, state)) ->
  (Tensor (n ': output) a, state)
forward step (parameters, (inputs, initial)) =
  let unliftResult value = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (A.Array (ShapeToDIM (n ': output)) a))
      count = fromIntegral $ natVal $ Proxy @n
      condition value = let (i, _, _) = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (A.Array (ShapeToDIM (n ': output)) a)) in A.unit $ A.the i A.< A.constant count
      body value =
        let (i, current, outputs) = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (A.Array (ShapeToDIM (n ': output)) a))
            (output, next) = step parameters (item (A.the i) inputs) $ rebuild @Tensor $ unpack current
            Tensor nextOutputs = writeItem (A.the i) output (Tensor outputs :: Tensor (n ': output) a)
         in A.lift (A.map (+ 1) i, pack $ flatten @Tensor next, nextOutputs)
      (_, final, result) =
        unliftResult $
          A.awhile condition body $
            A.lift (A.unit (A.constant (0 :: Int)), pack $ flatten @Tensor initial, unTensor $ full @(n ': output) 0)
   in (Tensor result, rebuild @Tensor $ unpack final)

-- | Record post-step states for reverse traversal. No history is kept by 'forward'.
forwardCached ::
  forall n p state input output a.
  ( KnownNat n,
    DType a,
    Num a,
    Num (TensorElement a),
    Shape input,
    Shape output,
    Shape (n ': input),
    Shape (n ': output),
    Flatten Tensor state,
    ScanValues a (Leaves Tensor state)
  ) =>
  (p -> Tensor input a -> state -> (Tensor output a, state)) ->
  (p, (Tensor (n ': input) a, state)) ->
  ((Tensor (n ': output) a, state), A.Acc (History (Leaves Tensor state)))
forwardCached step (parameters, (inputs, initial)) =
  let unliftResult value = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (A.Array (ShapeToDIM (n ': output)) a), A.Acc (History (Leaves Tensor state)))
      count = fromIntegral $ natVal $ Proxy @n
      condition value = let (i, _, _, _) = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (A.Array (ShapeToDIM (n ': output)) a), A.Acc (History (Leaves Tensor state))) in A.unit $ A.the i A.< A.constant count
      body value =
        let (i, current, outputs, previousHistory) = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (A.Array (ShapeToDIM (n ': output)) a), A.Acc (History (Leaves Tensor state)))
            (output, next) = step parameters (item (A.the i) inputs) $ rebuild @Tensor $ unpack current
            nextFields = flatten @Tensor next
            Tensor nextOutputs = writeItem (A.the i) output (Tensor outputs :: Tensor (n ': output) a)
         in A.lift (A.map (+ 1) i, pack nextFields, nextOutputs, saveHistory @a (A.the i) nextFields previousHistory)
      (_, final, result, history) =
        unliftResult $
          A.awhile condition body $
            A.lift (A.unit (A.constant (0 :: Int)), pack $ flatten @Tensor initial, unTensor $ full @(n ': output) 0, emptyHistory @a @(Leaves Tensor state) count)
   in ((Tensor result, rebuild @Tensor $ unpack final), history)
