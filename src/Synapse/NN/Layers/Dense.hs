{- | Provides dense layer implementation.

@Dense@ datatype represents densely-connected neural network layer and
it performs following operation: @x `matMul` w + b@, where @w@ is weights and @b@ is bias (if present) of a layer.
-}


{- @TypeFamilies@ are needed to instantiate @DType@.
-}

{-# LANGUAGE TypeFamilies #-}


module Synapse.NN.Layers.Dense
    ( -- * @Dense@ datatype

      Dense (Dense, denseWeights, denseBias, denseConstraints, denseRegularizers)
    , layerDenseWith
    , layerDense
    ) where


import Synapse.NN.Layers.Layer (AbstractLayer(..), LayerConfiguration)
import Synapse.NN.Layers.Initializers (Initializer(Initializer), zeroes, ones)
import Synapse.NN.Layers.Constraints (Constraint(Constraint))
import Synapse.NN.Layers.Regularizers (Regularizer(Regularizer))

import Synapse.Autograd (SymbolMat, Symbolic, unSymbol, symbol)

import Synapse.Tensors (DType, SingletonOps(singleton), MatOps(addMatRow, matMul))

import Synapse.Tensors.Vec (Vec)

import Synapse.Tensors.Mat (Mat)
import qualified Synapse.Tensors.Mat as M


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
weightsSymbol :: String -> Mat a -> SymbolMat a
weightsSymbol prefix = symbol (prefix ++ "1")


-- | Creates symbol for bias.
biasSymbol :: String -> Vec a -> SymbolMat a
biasSymbol prefix = symbol (prefix ++ "2") . M.rowVec

type instance DType (Dense a) = a

instance AbstractLayer Dense where
    inputSize = Just . M.nRows . denseWeights
    outputSize = Just . M.nCols . denseWeights

    nParameters _ = 2
    getParameters prefix (Dense weights bias _ _) = [weightsSymbol prefix weights, biasSymbol prefix bias]
    updateParameters (Dense _ _ constraints@(Constraint weightsConstraintFn, Constraint biasConstraintFn) regularizers) [weights', biasMat'] =
        Dense (weightsConstraintFn weights') (M.indexRow (biasConstraintFn biasMat') 0) constraints regularizers
    updateParameters _ _ = error "Parameters update failed - wrong amount of parameters was given"

    symbolicForward prefix input (Dense weights bias _ (Regularizer weightsRegularizerFn, Regularizer biasRegularizerFn)) =
        let symbolWeights = weightsSymbol prefix weights
            symbolBias = biasSymbol prefix bias
        in
        ( (input `matMul` symbolWeights) `addMatRow` symbolBias
        , weightsRegularizerFn symbolWeights + biasRegularizerFn symbolBias
        )

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

-- | Creates default configuration of dense layer - no constraints and weight are initialized with ones, bias is initialized with zeroes.
layerDense :: Symbolic a => Int -> LayerConfiguration (Dense a)
layerDense = layerDenseWith (Initializer ones, Constraint id, Regularizer (const $ singleton 0))
                            (Initializer zeroes, Constraint id, Regularizer (const $ singleton 0))
