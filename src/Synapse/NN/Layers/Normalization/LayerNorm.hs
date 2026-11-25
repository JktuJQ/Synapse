{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Normalization over trailing tensor dimensions.
module Synapse.NN.Layers.Normalization.LayerNorm (LayerNorm (..), layerNorm) where

import Control.Arrow (Arrow, arr)
import Data.Array.Accelerate qualified as A
import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.Generics (Generic)
import GHC.TypeLits (Nat)
import Synapse.Tensor.Ops.Constraints (FloatingDType)
import Synapse.Tensor.Ops.Definition (TensorOps (..), withCachedVJP)
import Synapse.Tensor.Ops.Raw.Reduction qualified as RawReduction
import Synapse.Tensor.Ops.Reduction qualified as R
import Synapse.Tensor.Ops.Shape qualified as T
import Synapse.Tensor.Representation (TensorLike)
import Synapse.Tensor.Representation qualified as Tensor
import Synapse.Tensor.Shapes.Axes (Axes, CanReduce, NonEmpty, ReduceShape)
import Synapse.Tensor.Shapes.Broadcast (CanBroadcast)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Synapse.Tensor.Shapes.Normalization (LayerNormAxes)
import Synapse.Tensor.Structure (TensorStructure, WithTensor)

-- | Per-element scale and bias over the trailing normalized dimensions.
data LayerNorm (normalized :: [Nat]) (t :: TensorLike) a = LayerNorm
    { layerNormWeights :: t normalized a
    , layerNormBias :: t normalized a
    }
    deriving (Generic)

instance TensorStructure (LayerNorm normalized Tensor.Tensor a) where
    type WithTensor t (LayerNorm normalized Tensor.Tensor a) = LayerNorm normalized t a

{- | Normalize trailing dimensions with population variance, then apply per-element scale and bias.

Epsilon must be positive.
-}
layerNorm
    :: forall normalized sh t a xs {arrow}
     . ( Arrow arrow
       , xs ~ LayerNormAxes normalized sh
       , TensorOps t a
       , RealFloat a
       , FloatingDType a
       , Floating (t sh a)
       , Fractional (t (ReduceShape xs sh) a)
       , Shape sh
       , Shape normalized
       , NonEmpty normalized
       , Shape (ReduceShape xs sh)
       , Axes xs
       , CanReduce xs sh
       , CanBroadcast normalized sh
       )
    => Double
    -> LayerNorm normalized t a
    -> arrow (t sh a) (t sh a)
layerNorm epsilon parameters = arr $ \input ->
    withCachedVJP
        (Proxy @(() :: Constraint))
        forward
        backward
        (input, layerNormWeights parameters, layerNormBias parameters)
  where
    expand = RawReduction.expandReduced @xs @sh
    forward (input, gamma, beta) =
        let mean = R.meanAxes @xs input
            centered = input - expand mean
            variance = R.meanAxes @xs (centered * centered)
            inv = recip $ sqrt $ variance + realToFrac epsilon
            normalized = centered * expand inv
         in (normalized * T.broadcast gamma + T.broadcast beta, (normalized, inv, gamma))
    backward (normalized, inv, gamma) seed =
        let weighted = seed * T.broadcast gamma
            dx =
                expand inv
                    * (weighted - expand (R.meanAxes @xs weighted) - normalized * expand (R.meanAxes @xs (weighted * normalized)))
         in (dx, sumParameters gamma (seed * normalized), sumParameters gamma seed)
    sumParameters
        :: Tensor.Tensor normalized a
        -> Tensor.Tensor sh a
        -> Tensor.Tensor normalized a
    sumParameters (Tensor.Tensor gamma) (Tensor.Tensor values) =
        Tensor.Tensor $
            A.reshape (A.shape gamma) $
                A.fold plus zero $
                    A.transpose $
                        A.reshape (A.constant $ A.Z A.:. shapeSize @(ReduceShape xs sh) A.:. shapeSize @normalized) values
      where
        zero = Tensor.unTensorElement (0 :: Tensor.TensorElement a)
        plus :: A.Exp a -> A.Exp a -> A.Exp a
        plus x y = Tensor.unTensorElement (Tensor.TensorElement x + Tensor.TensorElement y)
