{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Scatter selected-element and slice seeds back into the input shape.
module Synapse.Autograd.Tensor.Ops.Primitives.Index (index, indexAt, dynamicIndexOr, gather, gatherReduced, slice) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat)
import Synapse.Autograd.Tensor.Ops.Primitives.Shape (shapeOf)
import qualified Synapse.Tensor.Ops.Primitives.Index as Raw
import Synapse.Tensor.Shapes.Axes (RemoveAxis)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendCoordinates, backendShape, strides)
import Synapse.Tensor.Shapes.Index (CanGather, Index, Slice, SliceAxis (..), coordinates, sliceAxes)
import Synapse.Tensor.Type (DType)
import qualified Synapse.Tensor.Type as T

-- | Scatter a scalar seed to one statically selected input element.
index :: forall ix sh a proxy. (Index ix, Shape sh, DType a, Num (T.TensorElement a)) => proxy ix -> T.Tensor sh a -> T.TensorElement a -> T.TensorElement a -> T.Tensor sh a
index _ reference@(T.Tensor input) _ (T.TensorElement gradient) =
  let offset = sum (zipWith (*) (coordinates @ix) (strides (shapeOf reference)))
   in T.Tensor $ A.generate (A.shape input) $ \ix ->
        A.cond (A.toIndex (A.shape input) ix A.== A.constant offset) gradient (T.unTensorElement 0)

-- | Scatter a scalar seed to an element selected by a complete index expression.
indexAt :: (Shape sh, DType a, Num (T.TensorElement a)) => T.TensorIndex sh -> T.Tensor sh a -> T.TensorElement a -> T.TensorElement a -> T.Tensor sh a
indexAt (T.TensorIndex selected) (T.Tensor input) _ (T.TensorElement gradient) =
  T.Tensor $ A.generate (A.shape input) $ \ix ->
    A.cond (A.toIndex (A.shape input) ix A.== A.toIndex (A.shape input) selected) gradient (T.unTensorElement 0)

-- | Return a zero gradient for an invalid dynamic index.
dynamicIndexOr :: (Shape sh, DType a, Num (T.TensorElement a)) => T.TensorIndex sh -> T.Tensor sh a -> T.TensorElement a -> T.TensorElement a -> T.Tensor sh a
dynamicIndexOr selected input output (T.TensorElement gradient) =
  indexAt selected input output (T.TensorElement (A.cond (Raw.indexInBounds selected) gradient (T.unTensorElement 0)))

-- | Accumulate contributions at their selected source coordinates.
-- Checked gathers discard invalid selections before scattering.
gather :: forall axis sh indices a. (KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices, DType a, Num (T.TensorElement a)) => Bool -> T.Tensor indices Int -> T.Tensor sh a -> T.Tensor indices a -> T.Tensor indices a -> T.Tensor sh a
gather checked indices _ _ (T.Tensor gradient) =
  T.Tensor $
    A.permute (\x y -> T.unTensorElement (T.TensorElement x + T.TensorElement y)) zeros destination gradient
  where
    zeros = A.fill (A.constant (backendShape (Proxy @sh))) (T.unTensorElement (0 :: T.TensorElement a))
    destination ix =
      let selected@(T.TensorIndex coordinate) = Raw.gatherIndex @axis @sh indices ix
       in if checked then A.cond (Raw.indexInBounds selected) (A.Just_ coordinate) A.Nothing_ else A.Just_ coordinate

-- | Map source coordinates back into a slice, filling unselected positions with zero.
slice :: forall spec from to a proxy. (Slice spec, Shape from, Shape to, DType a, Num (T.TensorElement a)) => proxy spec -> T.Tensor from a -> T.Tensor to a -> T.Tensor to a -> T.Tensor from a
slice _ (T.Tensor reference) _ (T.Tensor gradient) = T.Tensor $ A.generate (A.shape reference) $ \ix ->
  let (inside, selected) = select (sliceAxes @spec) (backendCoordinates @from ix)
      offset = sum (zipWith (*) selected (map A.constant (strides (shapeOf (T.Tensor gradient :: T.Tensor to a)))))
   in A.cond inside (A.flatten gradient A.! A.index1 offset) (T.unTensorElement (0 :: T.TensorElement a))
  where
    select [] [] = (A.constant True, [])
    select (descriptor : rest) (coordinate : remaining) =
      let (inside, selected) = select rest remaining
          range begin stop step reversed =
            let delta = coordinate - A.constant begin
                count = if stop <= begin then 0 else 1 + (stop - begin - 1) `quot` step
                position = delta `A.quot` A.constant step
                output = if reversed then A.constant (count - 1) - position else position
                valid = coordinate A.>= A.constant begin A.&& coordinate A.< A.constant stop A.&& delta `A.rem` A.constant step A.== 0
             in (inside A.&& valid, output : selected)
       in case descriptor of
            All -> (inside, coordinate : selected)
            At fixed -> (inside A.&& coordinate A.== A.constant fixed, selected)
            Range begin stop step -> range begin stop step False
            RangeReverse begin stop step -> range begin stop step True
    select _ _ = error "Synapse.Autograd.slice: inconsistent slice rank"

-- | Scatter one class contribution per observation into the original class axis.
gatherReduced :: forall axis sh a. (KnownNat axis, Shape sh, Shape (RemoveAxis axis sh), DType a, Num (T.TensorElement a)) => T.Tensor (RemoveAxis axis sh) Int -> T.Tensor sh a -> T.Tensor (RemoveAxis axis sh) a -> T.Tensor (RemoveAxis axis sh) a -> T.Tensor sh a
gatherReduced indices _ _ (T.Tensor gradient) =
  T.Tensor $
    A.permute (\x y -> T.unTensorElement (T.TensorElement x + T.TensorElement y)) zeros destination gradient
  where
    zeros = A.fill (A.constant (backendShape (Proxy @sh))) (T.unTensorElement (0 :: T.TensorElement a))
    destination ix = let T.TensorIndex selected = Raw.gatherReducedIndex @axis @sh indices ix in A.Just_ selected
