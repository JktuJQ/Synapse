{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

-- | Patch extraction, pooling, and convolutions.
module Synapse.Tensor.Ops.Raw.Window
    ( -- * Window geometry
      Window (..)
    , window
    , coordinates
    , offset
    , windowLocation

      -- * Window matrices
    , extract
    , combine
    , patchesToMatrix
    , matrixToPatches

      -- * Patch tensors
    , extractPatches
    , combinePatches

      -- * Convolution geometry
    , convolutionWindow
    , groupCount

      -- * Convolution
    , convolve
    , transposeConvolve
    , transposeMatrix
    , outputMatrix
    , convolution
    , convolutionCached
    , convTranspose

      -- * Pooling geometry and contributions
    , poolWindow
    , poolLocation
    , maxWindows
    , nonOverlapping
    , divisors

      -- * Pooling
    , maxPool
    , avgPool
    ) where

import Data.List (zipWith4)
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat, natVal)
import Prelude

import Synapse.Tensor.Ops.Raw.Math (contract, matrixProduct)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Backend (backendShape, strides)
import Synapse.Tensor.Shapes.Dimensions (Shape (..))

import Data.Array.Accelerate qualified as A
import Data.Array.Accelerate.Representation.Type qualified as R
import Data.Array.Accelerate.Sugar.Elt qualified as Sugar
import Data.Array.Accelerate.Type qualified as AT
import Prelude qualified as P

import Synapse.Tensor.Ops.Logic qualified as E

{- | Spatial window geometry in channels-first order.
Strides, dilation, and padding ranks must already be validated by the caller.
Only before-padding is stored; both sides contribute to the precomputed output sizes.
-}
data Window = Window
    { batches :: Int
    , channels :: Int
    , inputSizes :: [Int]
    , kernelSizes :: [Int]
    , steps :: [Int]
    , spacing :: [Int]
    , beforePadding :: [Int]
    , outputSizes :: [Int]
    }
    deriving stock (Show)

-- | Reflect a validated channels-first input and its window parameters.
window
    :: [Int]
    -> [Int]
    -> [Int]
    -> [Int]
    -> [Int]
    -> Window
window (batch : cin : spatial) kernel stride padding dilation =
    Window batch cin spatial kernel stride dilation before $
        zipWith4 (\size k s (lo, hi) -> 1 + (size + lo + hi - k) `div` s) spatial effective stride pairs
  where
    effective = zipWith (\k d -> d * (k - 1) + 1) kernel dilation
    pairs = pair padding
    before = map fst pairs
    pair [] = []
    pair (lo : hi : rest) = (lo, hi) : pair rest
    pair _ = error "Invalid window padding rank"
window _ _ _ _ _ = error "Invalid window input rank"

-- | Decode a linear index using a host-known dimension list.
coordinates
    :: [Int]
    -> A.Exp Int
    -> [A.Exp Int]
coordinates dims linear = zipWith (\size stride -> linear `A.quot` A.constant stride `A.rem` A.constant size) dims $ strides dims

-- | Flatten row-major coordinates.
offset
    :: [Int]
    -> [A.Exp Int]
    -> A.Exp Int
offset dims = sum . zipWith (\stride coordinate -> A.constant stride * coordinate) (strides dims)

-- | Map a window column and flattened channel/kernel coordinate to a safe source offset.
windowLocation
    :: Window
    -> A.Exp Int
    -> A.Exp Int
    -> (A.Exp Bool, A.Exp Int)
windowLocation dims position inner = (valid, source)
  where
    spatial = product $ outputSizes dims
    volume = product $ kernelSizes dims
    b = position `A.quot` A.constant spatial
    c = inner `A.quot` A.constant volume
    output = coordinates (outputSizes dims) $ position `A.rem` A.constant spatial
    kernel = coordinates (kernelSizes dims) $ inner `A.rem` A.constant volume
    input =
        zipWith4
            (\o s k p -> o * A.constant s + k - A.constant p)
            output
            (steps dims)
            (zipWith (\k d -> k * A.constant d) kernel $ spacing dims)
            (beforePadding dims)
    valid =
        foldr (A.&&) (A.constant True) $
            zipWith (\size i -> i A.>= 0 A.&& i A.< A.constant size) (inputSizes dims) input
    safe = zipWith (\size i -> A.max 0 $ A.min (A.constant $ size - 1) i) (inputSizes dims) input
    source =
        (b * A.constant (channels dims) + c) * A.constant (product $ inputSizes dims) + offset (inputSizes dims) safe

-- | Extract a contraction-by-position matrix, using zero outside the input.
extract
    :: forall a
     . (DType a, Num (TensorElement a), P.Num a)
    => Window
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Matrix a)
extract dims input
    | 0 `elem` inputSizes dims = A.fill extent zero
    | interiorWindow dims = extractInterior dims input
    | otherwise = A.generate extent $ \ix ->
        let (valid, source) = windowLocation dims (A.indexHead ix) (A.indexHead $ A.indexTail ix)
         in A.cond valid (input A.! A.index1 source) zero
  where
    extent =
        A.constant $
            A.Z
                A.:. (channels dims * product (kernelSizes dims))
                A.:. (batches dims * product (outputSizes dims))
    zero = unTensorElement (0 :: TensorElement a)

interiorWindow :: Window -> Bool
interiorWindow dims =
    all (== 0) (beforePadding dims)
        && and
            ( zipWith4
                fits
                (inputSizes dims)
                (outputSizes dims)
                (kernelSizes dims)
                (zip (steps dims) (spacing dims))
            )
  where
    fits size outputs kernel (stride, dilation) =
        outputs > 0 && kernel > 0 && (outputs - 1) * stride + (kernel - 1) * dilation < size

-- | Separate the stride-dependent window bases from the dilated kernel offsets.
extractInterior
    :: DType a
    => Window
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Matrix a)
extractInterior dims input = A.generate (A.constant $ A.Z A.:. shared A.:. positions) $ \ix ->
    let k = A.indexHead (A.indexTail ix)
        p = A.indexHead ix
     in input A.! A.index1 (bases A.! A.index1 p + offsets A.! A.index1 k)
  where
    spatial = product (outputSizes dims)
    plane = product (inputSizes dims)
    volume = product (kernelSizes dims)
    positions = batches dims * spatial
    shared = channels dims * volume
    bases = A.compute $ A.generate (A.index1 $ A.constant positions) $ \ix ->
        let p = A.unindex1 ix
            b = p `A.quot` A.constant spatial
            coords = coordinates (outputSizes dims) (p `A.rem` A.constant spatial)
            shifted = zipWith (*) coords (map A.constant $ steps dims)
         in b * A.constant (channels dims * plane) + offset (inputSizes dims) shifted
    offsets =
        A.use $
            A.fromList
                (A.Z A.:. shared)
                [ c * plane + sum (zipWith3 (\x d stride -> x * d * stride) coords (spacing dims) (strides $ inputSizes dims))
                | c <- [0 .. channels dims - 1]
                , coords <- traverse (\k -> [0 .. k - 1]) (kernelSizes dims)
                ]

-- | Sum patch contributions for each input element by gathering its overlapping windows.
combine
    :: forall a
     . (DType a, Num (TensorElement a), P.Num a)
    => Window
    -> A.Acc (A.Matrix a)
    -> A.Acc (A.Vector a)
combine dims patches = contract count volume $ \linear inner ->
    let plane = linear `A.quot` A.constant inputSpatial
        b = plane `A.quot` A.constant (channels dims)
        c = plane `A.rem` A.constant (channels dims)
        input = coordinates (inputSizes dims) $ linear `A.rem` A.constant inputSpatial
        kernel = coordinates (kernelSizes dims) inner
        shifted = zipWith4 (\i p k d -> i + A.constant p - k * A.constant d) input (beforePadding dims) kernel (spacing dims)
        output = zipWith (\i s -> i `A.quot` A.constant s) shifted (steps dims)
        valid =
            foldr (A.&&) (A.constant True) $
                zipWith4
                    (\i s o size -> i A.>= 0 A.&& i `A.rem` A.constant s A.== 0 A.&& o A.< A.constant size)
                    shifted
                    (steps dims)
                    output
                    (outputSizes dims)
        safe = zipWith (\size o -> A.max 0 $ A.min (A.constant $ size - 1) o) (outputSizes dims) output
        position = b * A.constant outputSpatial + offset (outputSizes dims) safe
     in A.cond valid (patches A.! A.index2 (c * A.constant volume + inner) position) zero
  where
    inputSpatial = product $ inputSizes dims
    outputSpatial = product $ outputSizes dims
    count = batches dims * channels dims * inputSpatial
    volume = product $ kernelSizes dims
    zero = unTensorElement (0 :: TensorElement a)

-- | View public patches as a contraction-by-position matrix.
patchesToMatrix
    :: DType a
    => Window
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Matrix a)
patchesToMatrix dims patches = A.generate (A.constant $ A.Z A.:. shared A.:. (batches dims * spatial)) $ \ix ->
    let k = A.indexHead $ A.indexTail ix
        position = A.indexHead ix
        b = position `A.quot` A.constant spatial
        p = position `A.rem` A.constant spatial
     in patches A.! A.index1 ((b * A.constant shared + k) * A.constant spatial + p)
  where
    spatial = product $ outputSizes dims
    shared = channels dims * product (kernelSizes dims)

-- | Restore the public batch/channel-window/position layout.
matrixToPatches
    :: DType a
    => Window
    -> A.Acc (A.Matrix a)
    -> A.Acc (A.Array A.DIM3 a)
matrixToPatches dims matrix = A.generate (A.constant $ A.Z A.:. batches dims A.:. (channels dims * product (kernelSizes dims)) A.:. spatial) $ \ix ->
    let b = A.indexHead $ A.indexTail $ A.indexTail ix
        k = A.indexHead $ A.indexTail ix
        p = A.indexHead ix
     in matrix A.! A.index2 k (b * A.constant spatial + p)
  where
    spatial = product $ outputSizes dims

-- | Extract row-major spatial patches.
extractPatches
    :: forall kernel stride padding dilation input out a
     . ( Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape out
       , DType a
       , Num (TensorElement a)
       , P.Num a
       )
    => Tensor input a
    -> Tensor out a
extractPatches (Tensor input) =
    Tensor $
        A.reshape (A.constant $ backendShape $ Proxy @out) $
            matrixToPatches dims $
                extract dims $
                    A.flatten input
  where
    dims = window (shape @input) (shape @kernel) (shape @stride) (shape @padding) (shape @dilation)

-- | Sum overlapping patches into a specified output shape.
combinePatches
    :: forall output kernel stride padding dilation patches a
     . ( Shape output
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape patches
       , DType a
       , Num (TensorElement a)
       , P.Num a
       )
    => Tensor patches a
    -> Tensor output a
combinePatches (Tensor patches) =
    Tensor $
        A.reshape (A.constant $ backendShape $ Proxy @output) $
            combine dims $
                patchesToMatrix dims $
                    A.flatten patches
  where
    dims = window (shape @output) (shape @kernel) (shape @stride) (shape @padding) (shape @dilation)

-- Convolution

-- | Reflect the window geometry of a channels-first convolution.
convolutionWindow
    :: forall input kernel stride padding dilation
     . (Shape input, Shape kernel, Shape stride, Shape padding, Shape dilation)
    => Window
convolutionWindow = window (shape @input) (drop 2 $ shape @kernel) (shape @stride) (shape @padding) (shape @dilation)

-- | Reflect the positive, statically checked group count.
groupCount
    :: forall groups
     . KnownNat groups
    => Int
groupCount = fromIntegral $ natVal $ Proxy @groups

-- | Transpose each group matrix without expanding channels across groups.
transposeMatrix
    :: DType a
    => Int
    -> Int
    -> Int
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
transposeMatrix groups rows cols input =
    A.flatten $
        A.backpermute (A.constant $ A.Z A.:. groups A.:. cols A.:. rows) source $
            A.reshape (A.constant $ A.Z A.:. groups A.:. rows A.:. cols) input
  where
    source ix = A.index3 (A.indexHead $ A.indexTail $ A.indexTail ix) (A.indexHead ix) (A.indexHead $ A.indexTail ix)

-- | Reorder channels-first output into an output-channel-by-position matrix.
outputMatrix
    :: DType a
    => Window
    -> Int
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
outputMatrix dims cout input = A.generate (A.constant $ A.Z A.:. (cout * batches dims * spatial)) $ \ix ->
    let linear = A.unindex1 ix
        o = linear `A.quot` A.constant (batches dims * spatial)
        position = linear `A.rem` A.constant (batches dims * spatial)
        b = position `A.quot` A.constant spatial
        p = position `A.rem` A.constant spatial
     in input A.! A.index1 ((b * A.constant cout + o) * A.constant spatial + p)
  where
    spatial = product $ outputSizes dims

-- | Multiply kernels by windows and restore channels-first output order.
convolve
    :: (DType a, Num (TensorElement a), P.Num a)
    => Window
    -> Int
    -> Int
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
convolve dims cout groups input = convolveWindows dims cout groups (A.flatten $ extract dims input)

convolveWindows
    :: (DType a, Num (TensorElement a), P.Num a)
    => Window
    -> Int
    -> Int
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
convolveWindows dims cout groups windows weights = A.generate (A.constant $ A.Z A.:. (batches dims * cout * spatial)) $ \ix ->
    let linear = A.unindex1 ix
        position = (linear `A.quot` A.constant (cout * spatial)) * A.constant spatial + linear `A.rem` A.constant spatial
        o = (linear `A.quot` A.constant spatial) `A.rem` A.constant cout
     in result A.! A.index1 (o * A.constant positions + position)
  where
    spatial = product $ outputSizes dims
    positions = batches dims * spatial
    shared = channels dims `div` groups * product (kernelSizes dims)
    result =
        matrixProduct [groups] [groups] [groups] (cout `div` groups) shared positions weights windows

-- | Apply the transpose of a convolution's linear input mapping.
transposeConvolve
    :: (DType a, Num (TensorElement a), P.Num a)
    => Window
    -> Int
    -> Int
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
transposeConvolve dims cout groups weights input = combine dims $ A.reshape (A.constant $ A.Z A.:. (groups * shared) A.:. positions) patches
  where
    shared = channels dims `div` groups * product (kernelSizes dims)
    positions = batches dims * product (outputSizes dims)
    patches =
        matrixProduct
            [groups]
            [groups]
            [groups]
            shared
            (cout `div` groups)
            positions
            (transposeMatrix groups (cout `div` groups) shared weights)
            (outputMatrix dims cout input)

-- | Apply grouped cross-correlation with the validated output shape.
convolution
    :: forall stride padding dilation groups input kernel out a
     . ( Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape out
       , DType a
       , Num (TensorElement a)
       , P.Num a
       )
    => Tensor input a
    -> Tensor kernel a
    -> Tensor out a
convolution (Tensor input) (Tensor weights) =
    Tensor $
        A.reshape (A.constant $ backendShape $ Proxy @out) $
            convolve dims (product $ take 1 $ shape @kernel) (groupCount @groups) (A.flatten input) (A.flatten weights)
  where
    dims = convolutionWindow @input @kernel @stride @padding @dilation

-- | Retain the extracted windows for the kernel gradient.
convolutionCached
    :: forall stride padding dilation groups input kernel out a
     . ( Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape out
       , DType a
       , Num (TensorElement a)
       , P.Num a
       )
    => Tensor input a
    -> Tensor kernel a
    -> (Tensor out a, A.Acc (A.Vector a))
convolutionCached (Tensor input) (Tensor weights) =
    ( Tensor $
        A.reshape (A.constant $ backendShape $ Proxy @out) $
            convolveWindows dims (product $ take 1 $ shape @kernel) (groupCount @groups) windows (A.flatten weights)
    , windows
    )
  where
    dims = convolutionWindow @input @kernel @stride @padding @dilation
    windows = A.compute $ A.flatten $ extract dims $ A.flatten input

-- | Apply the transpose input mapping with the requested output shape.
convTranspose
    :: forall stride padding dilation groups input kernel out a
     . ( Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , KnownNat groups
       , Shape out
       , DType a
       , Num (TensorElement a)
       , P.Num a
       )
    => Tensor input a
    -> Tensor kernel a
    -> Tensor out a
convTranspose (Tensor input) (Tensor weights) =
    Tensor $
        A.reshape (A.constant $ backendShape $ Proxy @out) $
            transposeConvolve dims (shape @input !! 1) (groupCount @groups) (A.flatten weights) (A.flatten input)
  where
    dims = convolutionWindow @out @kernel @stride @padding @dilation

-- Pooling

-- | Reflect the validated geometry of a pooling operation.
poolWindow
    :: forall kernel stride padding input
     . (Shape kernel, Shape stride, Shape padding, Shape input)
    => [Int]
    -> Window
poolWindow = window (shape @input) (shape @kernel) (shape @stride) (shape @padding)

-- | Locate one kernel entry for a channels-first output element.
poolLocation
    :: Window
    -> A.Exp Int
    -> A.Exp Int
    -> (A.Exp Bool, A.Exp Int)
poolLocation dims row k = windowLocation dims (b * A.constant spatial + p) (c * A.constant volume + k)
  where
    spatial = product $ outputSizes dims
    volume = product $ kernelSizes dims
    b = row `A.quot` A.constant (channels dims * spatial)
    c = (row `A.quot` A.constant spatial) `A.rem` A.constant (channels dims)
    p = row `A.rem` A.constant spatial

-- | Padding contributes negative infinity for floats and the lower bound for integral elements.
paddingMinimum
    :: forall a
     . DType a
    => A.Exp a
paddingMinimum = A.constant $ Sugar.toElt @a $ lower $ Sugar.eltR @a
  where
    lower
        :: R.TupR AT.ScalarType b
        -> b
    lower R.TupRunit = ()
    lower (R.TupRpair x y) = (lower x, lower y)
    lower (R.TupRsingle (AT.SingleScalarType (AT.NumSingleType (AT.IntegralNumType t)))) = case AT.integralDict t of
        AT.IntegralDict -> minBound
    lower (R.TupRsingle (AT.SingleScalarType (AT.NumSingleType (AT.FloatingNumType t)))) = case AT.floatingDict t of
        AT.FloatingDict -> -(1 / 0)
    lower (R.TupRsingle (AT.VectorScalarType _)) = error "Max pooling requires scalar numeric elements"

-- | Check whether the pooling windows tile the input.
nonOverlapping :: Window -> Bool
nonOverlapping dims =
    all (> 0) (kernelSizes dims)
        && steps dims == kernelSizes dims
        && all (== 1) (spacing dims)
        && all (== 0) (beforePadding dims)
        && and (zipWith (\i k -> i `mod` k == 0) (inputSizes dims) (kernelSizes dims))
        && outputSizes dims == zipWith div (inputSizes dims) (kernelSizes dims)

-- | Select the first row-major maximum, or the first NaN, from valid input positions.
maxWindows
    :: forall a
     . (DType a, Num (TensorElement a), E.Ord (TensorElement a), P.Num a, P.Ord a)
    => Window
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector (Int, a))
maxWindows dims input
    | nonOverlapping dims && product (kernelSizes dims) <= 25 =
        A.generate (A.constant $ A.Z A.:. (batches dims * channels dims * product (outputSizes dims))) $ \ix ->
            let row = A.unindex1 ix
                plane = row `A.quot` A.constant (product $ outputSizes dims)
                position = row `A.rem` A.constant (product $ outputSizes dims)
                output = coordinates (outputSizes dims) position
                pick k =
                    let kernel = coordinates (kernelSizes dims) (A.constant k)
                        coords = zipWith3 (\o stride kx -> o * A.constant stride + kx) output (kernelSizes dims) kernel
                        source = plane * A.constant (product $ inputSizes dims) + offset (inputSizes dims) coords
                     in A.lift (A.constant True, A.constant k, input A.! A.index1 source)
             in discardValidity $ foldl choose (pick 0) [pick k | k <- [1 .. product (kernelSizes dims) - 1]]
    | otherwise = A.map discardValidity $ A.fold1 choose $ A.generate extent $ \ix ->
        let k = A.indexHead ix
            (valid, source) = poolLocation dims (A.indexHead $ A.indexTail ix) k
         in A.lift (valid, k, A.cond valid (input A.! A.index1 source) zero)
  where
    extent =
        A.constant $
            A.Z
                A.:. (batches dims * channels dims * product (outputSizes dims))
                A.:. product (kernelSizes dims)
    zero = paddingMinimum @a
    discardValidity value = let (valid, k, x) = A.unlift value :: (A.Exp Bool, A.Exp Int, A.Exp a) in A.lift (A.cond valid k (-1), x)
    choose lhs rhs =
        let (lv, li, lx) = A.unlift lhs :: (A.Exp Bool, A.Exp Int, A.Exp a)
            (rv, ri, rx) = A.unlift rhs :: (A.Exp Bool, A.Exp Int, A.Exp a)
            nanL = unTensorElement $ TensorElement lx E./= TensorElement lx
            nanR = unTensorElement $ TensorElement rx E./= TensorElement rx
            greater = unTensorElement $ TensorElement lx E.> TensorElement rx
            equal = unTensorElement $ TensorElement lx E.== TensorElement rx
            better =
                nanL A.&& (A.not nanR A.|| li A.<= ri)
                    A.|| A.not nanL A.&& A.not nanR A.&& (greater A.|| equal A.&& li A.<= ri)
         in A.cond (lv A.&& (A.not rv A.|| better)) lhs rhs

-- | Count the divisor for each output window in the tensor's numeric type.
divisors
    :: forall a
     . (DType a, Num (TensorElement a), P.Num a)
    => Bool
    -> Window
    -> A.Acc (A.Vector a)
divisors includePadding dims
    | includePadding =
        A.fill (A.constant $ A.Z A.:. count) $ unTensorElement (fromIntegral volume :: TensorElement a)
    | otherwise = contract count volume $ \row k ->
        A.cond
            (fst $ poolLocation dims row k)
            (unTensorElement (1 :: TensorElement a))
            (unTensorElement (0 :: TensorElement a))
  where
    count = batches dims * channels dims * product (outputSizes dims)
    volume = product $ kernelSizes dims

-- | Reduce each valid window to its maximum value.
maxPool
    :: forall kernel stride padding dilation input out a
     . ( Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape dilation
       , Shape out
       , DType a
       , Num (TensorElement a)
       , E.Ord (TensorElement a)
       , P.Num a
       , P.Ord a
       )
    => Tensor input a
    -> Tensor out a
maxPool (Tensor input) =
    Tensor $ A.reshape (A.constant $ backendShape $ Proxy @out) $ A.map A.snd $ maxWindows dims $ A.flatten input
  where
    dims = poolWindow @kernel @stride @padding @input (shape @dilation)

-- | Divide each window's sum by its full area or valid element count.
avgPool
    :: forall kernel stride padding input out a
     . ( Shape input
       , Shape kernel
       , Shape stride
       , Shape padding
       , Shape out
       , DType a
       , Fractional (TensorElement a)
       , P.Fractional a
       )
    => Bool
    -> Tensor input a
    -> Tensor out a
avgPool includePadding (Tensor input) =
    Tensor $
        A.reshape (A.constant $ backendShape $ Proxy @out) $
            A.zipWith divide totals $
                divisors includePadding dims
  where
    dims = poolWindow @kernel @stride @padding @input (replicate (shapeRank @kernel) 1)
    vector = A.flatten input
    totals = contract (batches dims * channels dims * product (outputSizes dims)) (product $ kernelSizes dims) $ \row k ->
        let (valid, source) = poolLocation dims row k
         in A.cond valid (vector A.! A.index1 source) (unTensorElement (0 :: TensorElement a))
    divide
        :: A.Exp a
        -> A.Exp a
        -> A.Exp a
    divide x y = unTensorElement $ TensorElement x / TensorElement y
