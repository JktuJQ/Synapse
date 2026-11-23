{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Model (Parameters, predict, initialParameters, emptyParameters) where

import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat)
import System.Random (mkStdGen)

import Synapse.NN qualified as NN
import Synapse.Tensor qualified as T

data Parameters t = Parameters
    { convolution :: NN.Convolution 1 4 '[5, 5] 1 t Double
    , output :: NN.Dense 576 10 t Double
    }
    deriving (Generic)

instance T.TensorStructure (Parameters T.Tensor) where
    type WithTensor t (Parameters T.Tensor) = Parameters t

predict
    :: forall batch t
     . (KnownNat batch, T.RealFloatTensorOps t Double)
    => Parameters t
    -> t '[batch, 1, 28, 28] Double
    -> t '[batch, 10] Double
predict parameters =
    NN.dense (output parameters)
        . T.reshape @'[batch, 576]
        . T.maxPool2d @'[2, 2] @'[2, 2] @'[0, 0, 0, 0] @'[1, 1]
        . NN.tanh
        . NN.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] (convolution parameters)

initialParameters :: Parameters T.Tensor
initialParameters = NN.evalState initialize (mkStdGen 17)
  where
    initialize = do
        filters <- NN.kaimingNormal @'[4, 1, 5, 5] 25 0
        weights <- NN.xavierUniform @'[576, 10] 576 10 1
        pure $ Parameters (NN.Convolution filters 0) (NN.Dense weights 0)

emptyParameters :: Parameters T.Tensor
emptyParameters = Parameters (NN.Convolution 0 0) (NN.Dense 0 0)
