{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}

module NumericPullbackDivision where

import qualified Synapse.Tensor as T
import Synapse.Tensor.Ops (Needs (Numeric), TensorOps (customVJP))

bad :: (TensorOps t a, T.Shape sh, Num (T.TensorElement a)) => t sh a -> t '[] a
bad = customVJP Numeric T.sum (\input -> (T.sum input, ())) (\() seed -> T.broadcast $ seed / 2)
