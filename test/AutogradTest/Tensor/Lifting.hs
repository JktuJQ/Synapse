{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoDeepSubsumption #-}

module AutogradTest.Tensor.Lifting (tests) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import Synapse.Tensor.Ops (TensorOps (..))
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: T.Shape sh => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

-- Ordinary cache metadata is not part of the differentiable tensor structure.
data Metadata = Metadata Int String

cachedSquare :: forall t sh a. (TensorOps t a, T.Shape sh, Num a, Num (T.TensorElement a)) => t sh a -> t sh a
cachedSquare =
    customVJP
        (Proxy @(() :: Constraint))
        (\x -> x * x)
        (\x -> (x * x, (x, Metadata 2 "derivative multiplier")))
        (\(x, Metadata multiplier _) seed -> fromIntegral multiplier * x * seed)

-- A user-defined reverse capability, independent of the standard numeric hierarchy.
class ReverseScale a where
    reverseScale :: T.Shape sh => T.Tensor sh a -> T.Tensor sh a

instance ReverseScale Double where
    reverseScale = (* 2)

customIdentity
    :: forall t sh a
     . (TensorOps t a, BackwardConstraints t (ReverseScale a), T.Shape sh)
    => t sh a
    -> t sh a
customIdentity = customVJP (Proxy @(ReverseScale a)) id (,()) (\() -> reverseScale)

data Parameters t = Parameters (t '[3] Double) (t '[3] Double) deriving (Generic)

instance T.TensorStructure (Parameters T.Tensor) where
    type WithTensor t (Parameters T.Tensor) = Parameters t

recordOperation :: forall t. TensorOps t Double => Parameters t -> (t '[] Double, t '[3] Double)
recordOperation =
    customVJP
        (Proxy @(() :: Constraint))
        (\(Parameters x y) -> (T.sum (x * y), x + y))
        (\p@(Parameters x y) -> ((T.sum (x * y), x + y), p))
        ( \(Parameters x y) (ds, dp) ->
            let common = T.broadcast ds
             in Parameters (common * y + dp) (common * x + dp)
        )

data OutputRecord t = OutputRecord (t '[] Double) (t '[3] Double) deriving (Generic)

instance T.TensorStructure (OutputRecord T.Tensor) where
    type WithTensor t (OutputRecord T.Tensor) = OutputRecord t

recordBoth :: forall t. TensorOps t Double => Parameters t -> OutputRecord t
recordBoth =
    customVJP
        (Proxy @(() :: Constraint))
        (\(Parameters x y) -> OutputRecord (T.sum $ x * y) (x + y))
        (\p@(Parameters x y) -> (OutputRecord (T.sum $ x * y) (x + y), p))
        ( \(Parameters x y) (OutputRecord ds dp) ->
            let common = T.broadcast ds
             in Parameters (common * y + dp) (common * x + dp)
        )

data Ten t
    = Ten
        (t '[1] Double)
        (t '[1] Double)
        (t '[1] Double)
        (t '[1] Double)
        (t '[1] Double)
        (t '[1] Double)
        (t '[1] Double)
        (t '[1] Double)
        (t '[1] Double)
        (t '[1] Double)
    deriving (Generic)

instance T.TensorStructure (Ten T.Tensor) where
    type WithTensor t (Ten T.Tensor) = Ten t

tenOperation :: forall t. TensorOps t Double => Ten t -> t '[1] Double
tenOperation =
    customVJP
        (Proxy @(() :: Constraint))
        (\(Ten a b c d e f g h i j) -> a + b + c + d + e + f + g + h + i + j)
        (\(Ten a b c d e f g h i j) -> (a + b + c + d + e + f + g + h + i + j, ()))
        (\() g -> Ten g g g g g g g g g g)

tenIdentity :: forall t. TensorOps t Double => Ten t -> Ten t
tenIdentity = customVJP (Proxy @(() :: Constraint)) id (,()) (\() gradient -> gradient)

threeOutputs
    :: TensorOps t Double => (t '[3] Double, t '[3] Double) -> (t '[] Double, (t '[3] Double, t '[3] Double))
threeOutputs =
    customVJP
        (Proxy @(() :: Constraint))
        (\(x, y) -> (T.sum $ x * y, (x + y, x - y)))
        (\(x, y) -> ((T.sum $ x * y, (x + y, x - y)), (x, y)))
        (\(x, y) (ds, (dp, dm)) -> (T.broadcast ds * y + dp + dm, T.broadcast ds * x + dp - dm))

tests :: TestTree
tests =
    testGroup
        "custom tensor primitives"
        [ testProperty "user-defined backward constraints do not restrict ordinary evaluation" $ H.withTests 1 $ H.property $ do
            let input = T.fromList @'[3] [False, True, False]
            T.toList (T.run T.CPU $ customIdentity input) H.=== [False, True, False]
        , testProperty "autograd supplies the constraint selected by each reverse rule" $ H.withTests 1 $ H.property $ do
            let input = T.fromList @'[3] [1, 2, 3 :: Double]
                (output, gradient) = AG.backprop customIdentity input
            values output H.=== [1, 2, 3]
            values gradient H.=== [2, 2, 2]
        , testProperty "private cache supports metadata and changing prepared inputs" $ H.withTests 1 $ H.property $ do
            let execute = T.run1 T.CPU $ \x -> AG.backprop cachedSquare (x :: T.Tensor '[3] Double)
                (out1, g1) = execute $ T.fromListData [1, 2, 3]
                (out2, g2) = execute $ T.fromListData [2, 4, 6]
            T.toList out1 H.=== [1, 4, 9]
            T.toList g1 H.=== [2, 4, 6]
            T.toList out2 H.=== [4, 16, 36]
            T.toList g2 H.=== [4, 8, 12]
        , testProperty "multiple results respect nonuniform seeds" $ H.withTests 1 $ H.property $ do
            let x = T.fromList @'[3] [1, 2, 3 :: Double]
                y = T.fromList @'[3] [4, 5, 6 :: Double]
                (_, backward) = AG.pullback threeOutputs (x, y)
                (dx, dy) = backward (2, (T.fromList [1, 2, 3], T.fromList [4, 5, 6]))
            values dx H.=== [13, 17, 21]
            values dy H.=== [-1, 1, 3]
        , testProperty "Generic records register independent inputs and zero unused results" $ H.withTests 1 $ H.property $ do
            let x = T.fromList @'[3] [1, 2, 3 :: Double]
                y = T.fromList @'[3] [4, 5, 6 :: Double]
                Parameters dx dy = AG.gradient (fst . recordOperation) (Parameters x y)
                (_, backward) = AG.pullback recordBoth (Parameters x y)
                Parameters dx2 dy2 = backward $ OutputRecord 2 (T.fromList [1, 2, 3])
            values dx H.=== [4, 5, 6]
            values dy H.=== [1, 2, 3]
            values dx2 H.=== [9, 12, 15]
            values dy2 H.=== [3, 6, 9]
        , testProperty "ten inputs and ten outputs have no arity limit" $ H.withTests 1 $ H.property $ do
            let input = Ten 1 2 3 4 5 6 7 8 9 10 :: Ten T.Tensor
                (out, Ten a b c d e f g h i j) = AG.backprop tenOperation input
                (_, Ten u v w x y z p q r s) = AG.backprop tenIdentity input
            values out H.=== [55]
            mapM_ (\gradient -> values gradient H.=== [1]) [a, b, c, d, e, f, g, h, i, j, u, v, w, x, y, z, p, q, r, s]
        , testProperty "shared input edges accumulate across all results" $ H.withTests 1 $ H.property $ do
            let input = T.fromList @'[3] [1, 2, 3 :: Double]
                (out, g) = AG.backprop (\x -> threeOutputs (x, x)) input
            values (fst out) H.=== [14]
            values g H.=== [4, 6, 8]
        , testProperty "ordinary Int evaluation skips both differentiation callbacks" $ H.withTests 1 $ H.property $ do
            let input = T.fromList @'[3] [1, 2, 3 :: Int]
                output =
                    customVJP
                        (Proxy @(Floating (T.TensorElement Int)))
                        T.sum
                        (error "cache evaluated")
                        (error "pullback evaluated")
                        input
            T.toList (T.run T.CPU output) H.=== [6]
        , testProperty "ordinary boolean inputs and type-changing results need no numeric instance" $ H.withTests 1 $ H.property $ do
            let input = T.fromList @'[3] [False, True, False]
                output = customVJP (Proxy @(() :: Constraint)) id (error "cache evaluated") (error "pullback evaluated") input
                integral = T.fromList @'[3] [0, 1, 2 :: Int]
                mask =
                    customVJP (Proxy @(() :: Constraint)) (T.> 1) (error "cache evaluated") (error "pullback evaluated") integral
            T.toList (T.run T.CPU output) H.=== [False, True, False]
            T.toList (T.run T.CPU mask) H.=== [False, False, True]
        , testProperty "pullback reuses the cached output" $ H.withTests 1 $ H.property $ do
            let operation :: TensorOps t Double => t '[2] Double -> t '[2] Double
                operation = customVJP (Proxy @(() :: Constraint)) exp (\x -> let y = exp x in (y, y)) (*)
                (out, g) = AG.backprop operation $ T.fromList [0, 1]
            approxList (values out) [1, exp 1]
            approxList (values g) [1, exp 1]
        ]
