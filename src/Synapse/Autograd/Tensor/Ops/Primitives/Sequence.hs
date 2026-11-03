{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Reverse traversal of a fixed-length scan using a local step pullback.
module Synapse.Autograd.Tensor.Ops.Primitives.Sequence (backward) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeNats (KnownNat, natVal)
import Synapse.Tensor.Backend.Structure (BackendTree (..))
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import Synapse.Tensor.Ops.Primitives.Construction (full)
import Synapse.Tensor.Ops.Primitives.Sequence (ScanValues (..), item, writeItem)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM)
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement)

-- | Accumulate parameter gradients and propagate the state cotangent backwards.
-- Step intermediates are recomputed from the saved pre-step state.
backward ::
  forall n p state input output a.
  ( KnownNat n,
    DType a,
    Num a,
    Num (TensorElement a),
    Shape input,
    Shape output,
    Shape (n ': input),
    Shape (n ': output),
    Flatten Tensor p,
    Flatten Tensor state,
    ScanValues a (Leaves Tensor p),
    ScanValues a (Leaves Tensor state)
  ) =>
  ((p, (Tensor input a, state)) -> (Tensor output a, state) -> (p, (Tensor input a, state))) ->
  ((p, (Tensor (n ': input) a, state)), A.Acc (History (Leaves Tensor state))) ->
  (Tensor (n ': output) a, state) ->
  (p, (Tensor (n ': input) a, state))
backward step ((parameters, (inputs, initial)), history) (seeds, finalSeed) =
  let unliftResult value = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (BackendLeaves (Leaves Tensor p)), A.Acc (A.Array (ShapeToDIM (n ': input)) a))
      count = fromIntegral $ natVal $ Proxy @n
      condition value = let (i, _, _, _) = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (BackendLeaves (Leaves Tensor p)), A.Acc (A.Array (ShapeToDIM (n ': input)) a)) in A.unit $ A.the i A.> 0
      body value =
        let (index, stateSeed, parameterSeed, inputSeeds) = A.unlift value :: (A.Acc (A.Scalar Int), A.Acc (BackendLeaves (Leaves Tensor state)), A.Acc (BackendLeaves (Leaves Tensor p)), A.Acc (A.Array (ShapeToDIM (n ': input)) a))
            i = A.the index - 1
            previous = A.acond (i A.== 0) (pack $ flatten @Tensor initial) $ pack $ readHistory @a @(Leaves Tensor state) (i - 1) history
            (dp, (dx, ds)) =
              step
                (parameters, (item i inputs, rebuild @Tensor $ unpack previous))
                (item i seeds, rebuild @Tensor $ unpack stateSeed)
            Tensor nextInputs = writeItem i dx (Tensor inputSeeds :: Tensor (n ': input) a)
         in A.lift
              ( A.map (subtract 1) index,
                pack $ flatten @Tensor ds,
                pack $ addValues @a (unpack parameterSeed) (flatten @Tensor dp),
                nextInputs
              )
      (_, stateGradient, parameterGradient, inputGradient) =
        unliftResult $
          A.awhile condition body $
            A.lift
              ( A.unit $ A.constant count,
                pack $ flatten @Tensor finalSeed,
                pack $ zeroValues @a @(Leaves Tensor p),
                unTensor $ full @(n ': input) 0
              )
   in (rebuild @Tensor $ unpack parameterGradient, (Tensor inputGradient, rebuild @Tensor $ unpack stateGradient))
