{- | Provides dense layer implementation.

@Dense@ datatype represents densely-connected neural network layer and
it performs following operation: @x `matMul` w + b@, where @w@ is weights and @b@ is bias (if present) of a layer.
-}


module Synapse.ML.Layers.Dense
    ( -- * @Dense@ datatype

      Dense (Dense, denseWeights, denseBias, denseConstraints, denseRegularizers)
    , layerDenseWith
    , layerDense
    ) where


import Synapse.ML.Layers.Layer (AbstractLayer(..), LayerConfiguration)
import Synapse.ML.Layers.Initializers (Initializer(Initializer), zeroes)
import Synapse.ML.Layers.Constraints (Constraint(Constraint))
import Synapse.ML.Layers.Regularizers (Regularizer(Regularizer))

import Synapse.Autograd (Symbol, Symbolic, symbol)

import Synapse.LinearAlgebra (Indexable(unsafeIndex), SingletonOps(singleton), MatOps(matMul))

import Synapse.LinearAlgebra.Vec (Vec)
import qualified Synapse.LinearAlgebra.Vec as V

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M


{- | @Dense@ datatype represents densely-connected neural network layer.

@Dense@ performs following operation: @x `matMul` w + b@, where @w@ is weights and @b@ is bias (if present) of a layer.
-}
data Dense a = Dense
    { denseWeights      :: Mat a                           -- ^ Matrix that represents weights of dense layer.
    , denseBias         :: Vec a                           -- ^ Vector that represents bias of dense layer.

    , denseConstraints  :: (Constraint a, Constraint a)    -- ^ Constraints on weights and bias of dense layer.
    , denseRegularizers :: (Regularizer a, Regularizer a)  -- ^ Regularizers on weights and bias of dense layer.
    }

-- | Creates symbol for weights.
weightsSymbol :: String -> Mat a -> Symbol (Mat a)
weightsSymbol prefix = symbol (prefix ++ "1")

-- | Creates matrix that corresponds to bias (bias rows stacked on each other).
biasToMat :: Int -> Vec a -> Mat a
biasToMat rows bias = M.generate (rows, V.size bias) $ \(_, c) -> unsafeIndex bias c

-- | Creates symbol for bias.
biasSymbol :: String -> Int -> Vec a -> Symbol (Mat a)
biasSymbol prefix rows = symbol (prefix ++ "2") . biasToMat rows

instance AbstractLayer Dense where
    inputSize = Just . M.nRows . denseWeights
    outputSize = Just . M.nCols . denseWeights

    getParameters (Dense weights bias _ _) = [weights, biasToMat (M.nRows weights) bias]
    updateParameters (Dense _ _ constraints@(Constraint weightsConstraintFn, Constraint biasConstraintFn) regularizers) [weights', biasMat'] =
        Dense (weightsConstraintFn weights') (M.indexRow (biasConstraintFn biasMat') 0) constraints regularizers
    updateParameters _ _ = error "Parameters update failed - wrong amount of parameters was given"

    applyRegularizer prefix (Dense weights bias _ (Regularizer weightsRegularizerFn, Regularizer biasRegularizerFn)) =
        weightsRegularizerFn (weightsSymbol prefix weights) + biasRegularizerFn (biasSymbol prefix (M.nRows weights) bias)

    symbolicForward prefix (Dense weights bias _ _) input =
        input `matMul` weightsSymbol prefix weights + biasSymbol prefix (M.nRows weights) bias

-- | Creates configuration of dense layer.
layerDenseWith
    :: Symbolic a
    => (Initializer a, Constraint a, Regularizer a)  -- ^ Weights initializer, constraint and regularizer.
    -> (Initializer a, Constraint a, Regularizer a)  -- ^ Bias initializer, constraint and regularizer.
    -> Int                              -- ^ Amount of neurons.
    -> LayerConfiguration (Dense a)
layerDenseWith (Initializer weightsInitializer, weightsConstraints, weightsRegularizer)
               (Initializer biasInitializer, biasConstraints, biasRegularizer)
               neurons input =
    Dense (weightsInitializer (input, neurons)) (M.indexRow (biasInitializer (1, neurons)) 0) 
          (weightsConstraints, biasConstraints) (weightsRegularizer, biasRegularizer)

-- | Creates default configuration of dense layer - no constraints and both weights and bias are initialized with zeroes.
layerDense :: Symbolic a => Int -> LayerConfiguration (Dense a)
layerDense = layerDenseWith (Initializer zeroes, Constraint id, Regularizer (const $ singleton 0))
                            (Initializer zeroes, Constraint id, Regularizer (const $ singleton 0))
