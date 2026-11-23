{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Model (Parameters, predict, initialParameters, emptyParameters) where

import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat)
import System.Random (mkStdGen)

import Synapse.NN qualified as NN
import Synapse.Tensor qualified as T

data Parameters t = Parameters
    { hidden :: NN.Dense 4 16 t Double
    , output :: NN.Dense 16 3 t Double
    }
    deriving (Generic)

instance T.TensorStructure (Parameters T.Tensor) where
    type WithTensor t (Parameters T.Tensor) = Parameters t

predict
    :: (KnownNat n, T.RealFloatTensorOps t Double)
    => Parameters t
    -> t '[n, 4] Double
    -> t '[n, 3] Double
predict parameters = NN.dense (output parameters) . NN.tanh . NN.dense (hidden parameters)

initialParameters :: Parameters T.Tensor
initialParameters = NN.evalState initialize (mkStdGen 17)
  where
    initialize = do
        first <- NN.xavierUniform @'[4, 16] 4 16 1
        second <- NN.xavierUniform @'[16, 3] 16 3 1
        pure $ Parameters (NN.Dense first 0) (NN.Dense second 0)

emptyParameters :: Parameters T.Tensor
emptyParameters = Parameters (NN.Dense 0 0) (NN.Dense 0 0)
