{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Floating-point conversions and their reverse conversions for gradient seeds.
module Synapse.Autograd.Tensor.Ops.Conversion () where

import Synapse.Autograd.Representation (Context, add)
import Synapse.Autograd.Tensor.Representation (Tensor (..), TensorElement (..))
import Synapse.Tensor.Ops.Constraints (NumDType)
import qualified Synapse.Tensor.Representation as T
import Synapse.Tensor.Shapes.Dimensions (Shape)

import qualified Numeric.Backprop.Explicit as BPE

import qualified Synapse.Tensor.Ops.Conversion as E

instance
    ( Context s
    , Shape sh
    , NumDType a
    , E.ToFloating (T.Tensor sh a) (T.Tensor sh b)
    , E.ToFloating (T.Tensor sh b) (T.Tensor sh a)
    )
    => E.ToFloating (Tensor s sh a) (Tensor s sh b)
    where
    toFloating (Tensor x) =
        Tensor $ BPE.liftOp1 (BPE.AF add) (BPE.op1 $ \v -> (E.toFloating v, E.toFloating)) x
instance
    ( Context s
    , NumDType a
    , E.ToFloating (T.TensorElement a) (T.TensorElement b)
    , E.ToFloating (T.TensorElement b) (T.TensorElement a)
    )
    => E.ToFloating (TensorElement s a) (TensorElement s b)
    where
    toFloating (TensorElement x) =
        TensorElement $ BPE.liftOp1 (BPE.AF add) (BPE.op1 $ \v -> (E.toFloating v, E.toFloating)) x
