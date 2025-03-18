{- | Provides learning rate functions - functions that return coefficient which modulates how big are updates of parameters in training.
-}


module Synapse.NN.LearningRates
    ( -- * 'LearningRateFn' type alias and 'LearningRate' newtype

      LearningRateFn
    , LearningRate (LearningRate, unLearningRate)

      -- * Learning rate decay functions
    
    , exponentialDecay
    , inverseTimeDecay
    , polynomialDecay
    , cosineDecay
    , piecewiseConstantDecay
    ) where


-- | 'LearningRateFn' type alias represents functions that return coefficient which modulates how big are updates of parameters in training.
type LearningRateFn a = Int -> a


-- | 'LearningRate' newtype wraps 'LearningRateFn's - functions that modulate how big are updates of parameters in training.
newtype LearningRate a = LearningRate
    { unLearningRate :: LearningRateFn a  -- ^ Unwraps 'LearningRate' newtype.
    }


-- Learning rate decay functions

-- | Takes initial learning rate, decay steps and decay rate and calculates exponential decay learning rate (@initial * decay_rate ^ (step / decay_steps)@).
exponentialDecay :: Num a => a -> Int -> a -> LearningRateFn a
exponentialDecay initial steps rate step = initial * rate ^ (step `div` steps)

-- | Takes initial learning rate, decay steps and decay rate and calculates inverse time decay learning rate (@initial / (1 + rate * step / steps))@).
inverseTimeDecay :: Fractional a => a -> Int -> a -> LearningRateFn a
inverseTimeDecay initial steps rate step = initial / (1.0 + rate * fromIntegral (step `div` steps))

{- | Takes initial learning rate, decay steps, polynomial power and end decay and calculates polynomial decay learning rate
(@if step < steps then initial * (1 - step / steps) ** power else end@).
-}
polynomialDecay :: Floating a => a -> Int -> a -> a -> LearningRateFn a
polynomialDecay initial steps power end step
    | step < steps = initial * fromIntegral (1 - min step steps `div` steps) ** power
    | otherwise    = end

{- | Takes initial learning rate, decay steps, alpha coefficient and warmup steps and target (optional) and calculates cosine decay learning rate
(@(1 - alpha) * (0.5 * (1.0 + cos (pi * step / steps))) + alpha) *
(if warmup then (if step < warmupSteps then (warmupLR - initial) * step / warmupSteps else warmupLR) else initial)@).
-}
cosineDecay :: Floating a => a -> Int -> a -> Maybe (Int, a) -> LearningRateFn a
cosineDecay initial steps alpha warmup step =
    ((1.0 - alpha) * (0.5 * (1.0 + cos (pi * fromIntegral (step `div` steps)))) + alpha) *
    case warmup of
        Nothing -> initial
        Just (warmupSteps, warmupLR) -> if step < warmupSteps
                                        then (warmupLR - initial) * fromIntegral (step `div` warmupSteps)
                                        else warmupLR

{- | Takes list of boundaries and learning rate values and last rate value for those boundaries and calculates piecewise constant decay learning rate
(@if step < bound1 then value1 else if step < bound2 then value2 else lastRate@ for @[(bound1, value1), (bound2, value2)]@).
-}
piecewiseConstantDecay :: [(Int, a)] -> a -> LearningRateFn a
piecewiseConstantDecay [] lastRate _ = lastRate
piecewiseConstantDecay ((stepBound, rateValue):xs) lastRate step
    | step < stepBound = rateValue
    | otherwise        = piecewiseConstantDecay xs lastRate step
