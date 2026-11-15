{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Reshaping, broadcasting, axis permutations, and tensor assembly.
module Synapse.Tensor.Ops.Shape
    ( -- * Shape transformations
      reshape
    , squeeze
    , unsqueeze
    , broadcast

      -- * Axis permutations
    , transpose
    , permuteAxes

      -- * Assembly
    , concat
    , stack
    ) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat)
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Shape as Pullback
import Synapse.Tensor.Ops.Definition (TensorOps (..))
import qualified Synapse.Tensor.Ops.Raw.Shape as Raw
import qualified Synapse.Tensor.Representation as T
import Synapse.Tensor.Shapes.Axes
    ( Axes
    , CanSqueeze
    , ConcatShape
    , InsertAxis
    , PermuteAxes
    , RemoveAxis
    , TransposeShape
    )
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Reshape (CanReshape)
import Prelude hiding (concat)

-- | Reshape a tensor without changing its row-major element order.
reshape
    :: forall {t} to from a
     . (TensorOps t a, Shape from, Shape to, CanReshape from to)
    => t from a
    -> t to a
reshape =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.reshape @to)
        (\primalInput -> let output = (Raw.reshape @to) primalInput in (output, (primalInput, output)))
        (uncurry Pullback.reshape)

-- | Remove a specified singleton axis.
squeeze
    :: forall {t} axis sh a
     . (TensorOps t a, Shape sh, Shape (RemoveAxis axis sh), CanSqueeze axis sh)
    => t sh a
    -> t (RemoveAxis axis sh) a
squeeze =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.reshape @(RemoveAxis axis sh))
        ( \primalInput -> let output = (Raw.reshape @(RemoveAxis axis sh)) primalInput in (output, (primalInput, output))
        )
        (uncurry Pullback.reshape)

-- | Insert a singleton dimension.
unsqueeze
    :: forall {t} axis sh a
     . (TensorOps t a, Shape sh, Shape (InsertAxis axis 1 sh))
    => t sh a
    -> t (InsertAxis axis 1 sh) a
unsqueeze =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.reshape @(InsertAxis axis 1 sh))
        ( \primalInput -> let output = (Raw.reshape @(InsertAxis axis 1 sh)) primalInput in (output, (primalInput, output))
        )
        (uncurry Pullback.reshape)

-- | Broadcast a tensor to a target shape.
broadcast
    :: forall {t} to from a
     . (TensorOps t a, Shape from, Shape to, CanBroadcast from to)
    => t from a
    -> t to a
broadcast =
    customVJP
        (Proxy @(Num (T.TensorElement a)))
        (Raw.broadcast @to)
        (\primalInput -> let output = (Raw.broadcast @to) primalInput in (output, (primalInput, output)))
        (uncurry Pullback.broadcast)

-- | Transpose the last two axes of a matrix or batched matrix.
transpose
    :: forall {t} sh a
     . (TensorOps t a, Shape sh, Shape (TransposeShape sh))
    => t sh a
    -> t (TransposeShape sh) a
transpose =
    customVJP
        (Proxy @(() :: Constraint))
        Raw.transpose
        (\primalInput -> let output = Raw.transpose primalInput in (output, (primalInput, output)))
        (uncurry Pullback.transpose)

-- | Reorder axes so each output axis corresponds to the specified input axis.
permuteAxes
    :: forall {t} xs sh a
     . (TensorOps t a, Axes xs, Shape sh, Shape (PermuteAxes xs sh))
    => t sh a
    -> t (PermuteAxes xs sh) a
permuteAxes =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.permuteAxes @xs Proxy)
        (\primalInput -> let output = Raw.permuteAxes @xs Proxy primalInput in (output, (primalInput, output)))
        (uncurry (Pullback.permuteAxes @xs Proxy))

-- | Concatenate two tensors along an existing zero-based axis.
concat
    :: forall {t} axis lhs rhs a
     . (TensorOps t a, KnownNat axis, Shape lhs, Shape rhs, Shape (ConcatShape axis lhs rhs))
    => t lhs a
    -> t rhs a
    -> t (ConcatShape axis lhs rhs) a
concat =
    curry $
        customVJP
            (Proxy @(() :: Constraint))
            (uncurry (Raw.concat @axis Proxy))
            ( \(left, right) ->
                let output = Raw.concat @axis Proxy left right
                 in (output, ((left, right), output))
            )
            ( \((left, right), output) ->
                Pullback.concat @axis Proxy left right output
            )

-- | Stack two equally shaped tensors by inserting a dimension of size two.
stack
    :: forall {t} axis sh a
     . (TensorOps t a, KnownNat axis, Shape sh, Shape (InsertAxis axis 2 sh))
    => t sh a
    -> t sh a
    -> t (InsertAxis axis 2 sh) a
stack =
    curry $
        customVJP
            (Proxy @(() :: Constraint))
            (uncurry (Raw.stack @axis Proxy))
            ( \(left, right) ->
                let output = Raw.stack @axis Proxy left right
                 in (output, ((left, right), output))
            )
            ( \((left, right), output) ->
                Pullback.stack @axis Proxy left right output
            )
