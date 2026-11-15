{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Spatial windows, convolutions, transposed convolutions, and pooling.
module Synapse.Tensor.Ops.Window
    ( -- * Patch extraction
      extractPatches1d
    , extractPatches2d
    , extractPatches3d

      -- * Overlap summation
    , combinePatches1d
    , combinePatches2d
    , combinePatches3d

      -- * Convolution
    , conv1d
    , conv2d
    , conv3d

      -- * Transposed convolution
    , convTranspose1d
    , convTranspose2d
    , convTranspose3d

      -- * Max pooling
    , maxPool1d
    , maxPool2d
    , maxPool3d

      -- * Average pooling
    , avgPool1d
    , avgPool2d
    , avgPool3d
    ) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat)

import Synapse.Tensor.Ops.Definition (TensorOps (customVJP))
import Synapse.Tensor.Representation (TensorElement)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Window
    ( Conv1dShape
    , Conv2dShape
    , Conv3dShape
    , ConvTranspose1dShape
    , ConvTranspose2dShape
    , ConvTranspose3dShape
    , Patches1dShape
    , Patches2dShape
    , Patches3dShape
    , Pool1dShape
    , Pool2dShape
    , Pool3dShape
    )

import qualified Synapse.Autograd.Tensor.Ops.Primitives.Convolution as ConvPullback
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Window as WindowPullback
import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Raw.Window as Raw

-- | Extract zero-padded 1D windows from NCL input as @[batch, channels * kernel_volume, positions]@.
extractPatches1d
    :: forall kernel stride padding dilation {t} input a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape (Patches1dShape input kernel stride padding dilation)
       )
    => t input a
    -> t (Patches1dShape input kernel stride padding dilation) a
extractPatches1d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.extractPatches @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.extractPatches @kernel @stride @padding @dilation) primalInput
             in (output, (primalInput, output))
        )
        (uncurry (WindowPullback.extractPatches @kernel @stride @padding @dilation))

-- | Extract zero-padded 2D windows from NCHW input as @[batch, channels * kernel_volume, positions]@.
extractPatches2d
    :: forall kernel stride padding dilation {t} input a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape (Patches2dShape input kernel stride padding dilation)
       )
    => t input a
    -> t (Patches2dShape input kernel stride padding dilation) a
extractPatches2d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.extractPatches @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.extractPatches @kernel @stride @padding @dilation) primalInput
             in (output, (primalInput, output))
        )
        (uncurry (WindowPullback.extractPatches @kernel @stride @padding @dilation))

-- | Extract zero-padded 3D windows from NCDHW input as @[batch, channels * kernel_volume, positions]@.
extractPatches3d
    :: forall kernel stride padding dilation {t} input a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape (Patches3dShape input kernel stride padding dilation)
       )
    => t input a
    -> t (Patches3dShape input kernel stride padding dilation) a
extractPatches3d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.extractPatches @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.extractPatches @kernel @stride @padding @dilation) primalInput
             in (output, (primalInput, output))
        )
        (uncurry (WindowPullback.extractPatches @kernel @stride @padding @dilation))

-- | Sum patches into a specified NCL output shape, dropping entries outside that shape.
combinePatches1d
    :: forall output kernel stride padding dilation {t} patches a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape output
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape patches
       , patches ~ Patches1dShape output kernel stride padding dilation
       )
    => t patches a
    -> t output a
combinePatches1d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.combinePatches @output @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.combinePatches @output @kernel @stride @padding @dilation) primalInput
             in (output, (primalInput, output))
        )
        (uncurry (WindowPullback.combinePatches @kernel @stride @padding @dilation))

-- | Sum patches into a specified NCHW output shape, dropping entries outside that shape.
combinePatches2d
    :: forall output kernel stride padding dilation {t} patches a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape output
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape patches
       , patches ~ Patches2dShape output kernel stride padding dilation
       )
    => t patches a
    -> t output a
combinePatches2d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.combinePatches @output @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.combinePatches @output @kernel @stride @padding @dilation) primalInput
             in (output, (primalInput, output))
        )
        (uncurry (WindowPullback.combinePatches @kernel @stride @padding @dilation))

-- | Sum patches into a specified NCDHW output shape, dropping entries outside that shape.
combinePatches3d
    :: forall output kernel stride padding dilation {t} patches a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape output
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape patches
       , patches ~ Patches3dShape output kernel stride padding dilation
       )
    => t patches a
    -> t output a
combinePatches3d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.combinePatches @output @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.combinePatches @output @kernel @stride @padding @dilation) primalInput
             in (output, (primalInput, output))
        )
        (uncurry (WindowPullback.combinePatches @kernel @stride @padding @dilation))

-- | Apply 1D cross-correlation to NCL input using OIK weights, without reversing the kernel.
conv1d
    :: forall stride padding dilation groups {t} input kernel a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape (Conv1dShape input kernel stride padding dilation groups)
       )
    => t input a
    -> t kernel a
    -> t (Conv1dShape input kernel stride padding dilation groups) a
conv1d =
    curry $
        customVJP
            (Proxy @(() :: Constraint))
            (uncurry (Raw.convolution @stride @padding @dilation @groups))
            ( \(left, right) ->
                let output =
                        (Raw.convolution @stride @padding @dilation @groups) left right
                 in (output, ((left, right), output))
            )
            ( \((left, right), output) ->
                (ConvPullback.convolution @stride @padding @dilation @groups)
                    left
                    right
                    output
            )

-- | Apply 2D cross-correlation to NCHW input using OIHW weights, without reversing the kernel.
conv2d
    :: forall stride padding dilation groups {t} input kernel a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape (Conv2dShape input kernel stride padding dilation groups)
       )
    => t input a
    -> t kernel a
    -> t (Conv2dShape input kernel stride padding dilation groups) a
conv2d =
    curry $
        customVJP
            (Proxy @(() :: Constraint))
            (uncurry (Raw.convolution @stride @padding @dilation @groups))
            ( \(left, right) ->
                let output =
                        (Raw.convolution @stride @padding @dilation @groups) left right
                 in (output, ((left, right), output))
            )
            ( \((left, right), output) ->
                (ConvPullback.convolution @stride @padding @dilation @groups)
                    left
                    right
                    output
            )

-- | Apply 3D cross-correlation to NCDHW input using OIDHW weights, without reversing the kernel.
conv3d
    :: forall stride padding dilation groups {t} input kernel a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape (Conv3dShape input kernel stride padding dilation groups)
       )
    => t input a
    -> t kernel a
    -> t (Conv3dShape input kernel stride padding dilation groups) a
conv3d =
    curry $
        customVJP
            (Proxy @(() :: Constraint))
            (uncurry (Raw.convolution @stride @padding @dilation @groups))
            ( \(left, right) ->
                let output =
                        (Raw.convolution @stride @padding @dilation @groups) left right
                 in (output, ((left, right), output))
            )
            ( \((left, right), output) ->
                (ConvPullback.convolution @stride @padding @dilation @groups)
                    left
                    right
                    output
            )

-- | Apply transposed 1D convolution to NCL input using IOK weights.
convTranspose1d
    :: forall stride padding dilation groups extra {t} input kernel a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape (ConvTranspose1dShape input kernel stride padding dilation groups extra)
       )
    => t input a
    -> t kernel a
    -> t (ConvTranspose1dShape input kernel stride padding dilation groups extra) a
convTranspose1d =
    curry $
        customVJP
            (Proxy @(() :: Constraint))
            (uncurry (Raw.convTranspose @stride @padding @dilation @groups))
            ( \(left, right) ->
                let output =
                        (Raw.convTranspose @stride @padding @dilation @groups) left right
                 in (output, ((left, right), output))
            )
            ( \((left, right), output) ->
                (ConvPullback.convTranspose @stride @padding @dilation @groups)
                    left
                    right
                    output
            )

-- | Apply transposed 2D convolution to NCHW input using IOHW weights.
convTranspose2d
    :: forall stride padding dilation groups extra {t} input kernel a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape (ConvTranspose2dShape input kernel stride padding dilation groups extra)
       )
    => t input a
    -> t kernel a
    -> t (ConvTranspose2dShape input kernel stride padding dilation groups extra) a
convTranspose2d =
    curry $
        customVJP
            (Proxy @(() :: Constraint))
            (uncurry (Raw.convTranspose @stride @padding @dilation @groups))
            ( \(left, right) ->
                let output =
                        (Raw.convTranspose @stride @padding @dilation @groups) left right
                 in (output, ((left, right), output))
            )
            ( \((left, right), output) ->
                (ConvPullback.convTranspose @stride @padding @dilation @groups)
                    left
                    right
                    output
            )

-- | Apply transposed 3D convolution to NCDHW input using IODHW weights.
convTranspose3d
    :: forall stride padding dilation groups extra {t} input kernel a
     . ( TensorOps t a
       , Num (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape (ConvTranspose3dShape input kernel stride padding dilation groups extra)
       )
    => t input a
    -> t kernel a
    -> t (ConvTranspose3dShape input kernel stride padding dilation groups extra) a
convTranspose3d =
    curry $
        customVJP
            (Proxy @(() :: Constraint))
            (uncurry (Raw.convTranspose @stride @padding @dilation @groups))
            ( \(left, right) ->
                let output =
                        (Raw.convTranspose @stride @padding @dilation @groups) left right
                 in (output, ((left, right), output))
            )
            ( \((left, right), output) ->
                (ConvPullback.convTranspose @stride @padding @dilation @groups)
                    left
                    right
                    output
            )

-- | Maximum over 1D spatial windows in NCL layout.
maxPool1d
    :: forall kernel stride padding dilation {t} input a
     . ( TensorOps t a
       , Num (TensorElement a)
       , E.Ord (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape (Pool1dShape input kernel stride padding dilation)
       )
    => t input a
    -> t (Pool1dShape input kernel stride padding dilation) a
maxPool1d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.maxPool @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.maxPool @kernel @stride @padding @dilation) primalInput in (output, (primalInput, output))
        )
        (uncurry (ConvPullback.maxPool @kernel @stride @padding @dilation))

-- | Maximum over 2D spatial windows in NCHW layout.
maxPool2d
    :: forall kernel stride padding dilation {t} input a
     . ( TensorOps t a
       , Num (TensorElement a)
       , E.Ord (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape (Pool2dShape input kernel stride padding dilation)
       )
    => t input a
    -> t (Pool2dShape input kernel stride padding dilation) a
maxPool2d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.maxPool @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.maxPool @kernel @stride @padding @dilation) primalInput in (output, (primalInput, output))
        )
        (uncurry (ConvPullback.maxPool @kernel @stride @padding @dilation))

-- | Maximum over 3D spatial windows in NCDHW layout.
maxPool3d
    :: forall kernel stride padding dilation {t} input a
     . ( TensorOps t a
       , Num (TensorElement a)
       , E.Ord (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape (Pool3dShape input kernel stride padding dilation)
       )
    => t input a
    -> t (Pool3dShape input kernel stride padding dilation) a
maxPool3d =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.maxPool @kernel @stride @padding @dilation)
        ( \primalInput ->
            let output = (Raw.maxPool @kernel @stride @padding @dilation) primalInput in (output, (primalInput, output))
        )
        (uncurry (ConvPullback.maxPool @kernel @stride @padding @dilation))

{- | Average over 1D spatial windows in NCL layout.
Include zero padding in the divisor when @includePadding@ is 'True'.
-}
avgPool1d
    :: forall kernel stride padding {t} input a
     . ( TensorOps t a
       , Fractional (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape (Pool1dShape input kernel stride padding '[1])
       )
    => Bool
    -> t input a
    -> t (Pool1dShape input kernel stride padding '[1]) a
avgPool1d includePadding =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.avgPool @kernel @stride @padding includePadding)
        ( \primalInput ->
            let output = Raw.avgPool @kernel @stride @padding includePadding primalInput
             in (output, (primalInput, output))
        )
        (uncurry (ConvPullback.avgPool @kernel @stride @padding includePadding))

{- | Average over 2D spatial windows in NCHW layout.
Include zero padding in the divisor when @includePadding@ is 'True'.
-}
avgPool2d
    :: forall kernel stride padding {t} input a
     . ( TensorOps t a
       , Fractional (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape (Pool2dShape input kernel stride padding '[1, 1])
       )
    => Bool
    -> t input a
    -> t (Pool2dShape input kernel stride padding '[1, 1]) a
avgPool2d includePadding =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.avgPool @kernel @stride @padding includePadding)
        ( \primalInput ->
            let output = Raw.avgPool @kernel @stride @padding includePadding primalInput
             in (output, (primalInput, output))
        )
        (uncurry (ConvPullback.avgPool @kernel @stride @padding includePadding))

{- | Average over 3D spatial windows in NCDHW layout.
Include zero padding in the divisor when @includePadding@ is 'True'.
-}
avgPool3d
    :: forall kernel stride padding {t} input a
     . ( TensorOps t a
       , Fractional (TensorElement a)
       , Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape (Pool3dShape input kernel stride padding '[1, 1, 1])
       )
    => Bool
    -> t input a
    -> t (Pool3dShape input kernel stride padding '[1, 1, 1]) a
avgPool3d includePadding =
    customVJP
        (Proxy @(() :: Constraint))
        (Raw.avgPool @kernel @stride @padding includePadding)
        ( \primalInput ->
            let output = Raw.avgPool @kernel @stride @padding includePadding primalInput
             in (output, (primalInput, output))
        )
        (uncurry (ConvPullback.avgPool @kernel @stride @padding includePadding))
