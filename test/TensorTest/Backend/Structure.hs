{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module TensorTest.Backend.Structure (tests) where

import Control.Monad.Trans.State.Strict (modify, runState)
import Data.Functor.Identity (Identity (..))
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Structure as Structure
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Inputs t = Inputs {numbers :: t '[2] Double, mask :: t '[2] Bool} deriving (Generic)

instance T.TensorStructure (Inputs T.Tensor) where
    type WithTensor t (Inputs T.Tensor) = Inputs t

newtype Nested t = Nested {layer :: NN.Dense 1 1 t Double} deriving (Generic)

instance T.TensorStructure (Nested T.Tensor) where
    type WithTensor t (Nested T.Tensor) = Nested t

data Positional t = Positional (Inputs t) (t '[] Int) (t '[] Bool)
    deriving (Generic)

instance Structure.TensorStructure (Positional T.Tensor) where
    type WithTensor t (Positional T.Tensor) = Positional t

class KeepFirst a where
    keepFirst :: a -> a -> a

instance KeepFirst (T.Tensor sh a) where
    keepFirst x _ = x

class Unrestricted (representations :: (T.TensorLike, T.TensorLike)) sh a
instance Unrestricted representations sh a

newtype WrappedTensor sh a = WrappedTensor (T.Tensor sh a)

class (T.Shape sh, T.DType a, Show a) => Readable (t :: T.TensorLike) sh a
instance (T.Shape sh, T.DType a, Show a) => Readable t sh a

class (T.Shape sh, T.DType a) => Materialize (representations :: (T.TensorLike, T.TensorLike)) sh a
instance (T.Shape sh, T.DType a) => Materialize '(T.Tensor, T.TensorData) sh a

class
    (representations ~ '(T.TensorData, T.Tensor, T.Tensor), sh ~ '[2], a ~ Double) =>
    VectorPair (representations :: (T.TensorLike, T.TensorLike, T.TensorLike)) sh a
instance VectorPair '(T.TensorData, T.Tensor, T.Tensor) '[2] Double

wrapUnknown :: forall sh a. T.Tensor sh a -> WrappedTensor sh a
wrapUnknown = Structure.map @Unrestricted @(T.Tensor sh a) @T.Tensor @WrappedTensor (\_ -> WrappedTensor)

step :: Nested T.Tensor -> T.Tensor '[] Double -> (T.Tensor '[] Double, Nested T.Tensor)
step parameters lr =
    let objective p = T.sum (NN.dense (layer p) (AG.constant (T.fromList @'[1] [2 :: Double])) ** 2)
        (loss, Nested gradients) = AG.backprop objective parameters
        NN.Dense w b = layer parameters
        updated =
            NN.Dense
                (w - T.broadcast @'[1, 1] lr * NN.denseWeights gradients)
                (b - T.broadcast @'[1] lr * NN.denseBias gradients)
     in (loss, Nested updated)

coefficients :: Nested T.TensorData -> [Double]
coefficients (Nested p) = T.toList (NN.denseWeights p) ++ T.toList (NN.denseBias p)

tests :: TestTree
tests =
    testGroup
        "structured execution"
        [ testProperty "flatten and rebuild preserve a nested layer" $ H.withTests 1 $ H.property $ do
            let original = Nested (NN.Dense (T.fromList @'[1, 1] [3]) (T.fromList @'[1] [2]))
                (weightsField, biasField) = Structure.flatten @(Nested T.Tensor) @T.Tensor original
                restored = Structure.rebuild @(Nested T.Tensor) @T.Tensor (weightsField, biasField)
                weights = Structure.tensorFieldValue weightsField
                bias = Structure.tensorFieldValue biasField
            Structure.tensorFieldPath weightsField H.=== ["layer", "denseWeights"]
            Structure.tensorFieldPath biasField H.=== ["layer", "denseBias"]
            T.toList (T.run T.CPU weights) H.=== [3]
            T.toList (T.run T.CPU bias) H.=== [2]
            coefficients (T.run T.CPU restored) H.=== [3, 2]
        , testProperty "paths preserve selectors nested positions and root tensors" $ H.withTests 1 $ H.property $ do
            let input =
                    Positional
                        (Inputs (T.fromListData @'[2] [2, 3]) (T.fromListData @'[2] [True, False]))
                        (T.fromListData @'[] [7])
                        (T.fromListData @'[] [False])
                paths =
                    Structure.foldElements @Show @(Positional T.Tensor) @T.TensorData
                        (\path acc _ -> acc ++ [path])
                        []
                        input
                root =
                    Structure.foldElements @Show @(T.Tensor '[] Int) @T.TensorData
                        (\path acc _ -> acc ++ [path])
                        []
                        (T.fromListData @'[] [1])
            paths H.=== [["0", "numbers"], ["0", "mask"], ["1"], ["2"]]
            root H.=== [[]]
        , testProperty "metadata paths do not inspect tensor values" $ H.withTests 1 $ H.property $ do
            let input = Nested (NN.Dense undefined undefined) :: Nested T.TensorData
                paths =
                    Structure.foldElements @Show @(Nested T.Tensor) @T.TensorData
                        (\path acc _ -> acc ++ [path])
                        []
                        input
            paths H.=== [["layer", "denseWeights"], ["layer", "denseBias"]]
        , testProperty "general and specialized traversals receive identical paths" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                visit path tensor = modify (++ [path]) >> pure tensor
                (_, general) =
                    runState
                        (Structure.traverse @Unrestricted @(Inputs T.Tensor) @T.Tensor @T.Tensor visit input)
                        []
                (_, fields) = runState (Structure.traverseFields @Show visit input) []
                (_, elements) =
                    runState
                        (Structure.traverseElements @Show @(Inputs T.Tensor) @T.Tensor @T.Tensor visit input)
                        []
            general H.=== [["numbers"], ["mask"]]
            fields H.=== general
            elements H.=== general
        , testProperty "map and zip can select parameters by path" $ H.withTests 1 $ H.property $ do
            let input = (T.full @'[] (2 :: Double), T.full @'[1] (3 :: Double))
                change path tensor = if path == ["0"] then negate tensor else tensor
                combine path left right = if path == ["0"] then left - right else left + right
                general =
                    Structure.map @Materialize @_ @T.Tensor @T.TensorData
                        (\path tensor -> if path == ["0"] || path == ["1"] then T.run T.CPU tensor else error "unexpected path")
                        input
                fields = Structure.mapFields @Num change input
                elements = Structure.mapElements @((~) Double) @_ @T.Tensor @T.Tensor change input
                zippedFields = Structure.zipWithFields @Num combine input input
                zippedElements = Structure.zipWithElements @((~) Double) @_ @T.Tensor @T.Tensor @T.Tensor combine input input
                values (left, right) = (T.toList left, T.toList right)
            values general H.=== ([2], [3])
            values (T.run T.CPU fields) H.=== ([-2], [3])
            values (T.run T.CPU elements) H.=== ([-2], [3])
            values (T.run T.CPU zippedFields) H.=== ([0], [6])
            values (T.run T.CPU zippedElements) H.=== ([0], [6])
        , testProperty "materialized products can be edited and rebuilt" $ H.withTests 1 $ H.property $ do
            let original = Inputs (T.fromListData @'[2] [2, 3]) (T.fromListData @'[2] [True, False])
                (values, _) = Structure.flatten @(Inputs T.Tensor) @T.TensorData original
                replacement = T.fromListData @'[2] [False, True]
                restored = Structure.rebuild @(Inputs T.Tensor) @T.TensorData (values, Structure.TensorField ["ignored"] replacement)
            T.toList (numbers restored) H.=== [2, 3]
            T.toList (mask restored) H.=== [False, True]
            let (_, restoredMask) = Structure.flatten @(Inputs T.Tensor) @T.TensorData restored
            Structure.tensorFieldPath restoredMask H.=== ["mask"]
        , testProperty "map changes representation in a mixed tensor record" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                Inputs (WrappedTensor numbers') (WrappedTensor mask') =
                    Structure.map @Unrestricted @(Inputs T.Tensor) @T.Tensor @WrappedTensor (\_ -> wrapUnknown) input
            T.toList (T.run T.CPU numbers') H.=== [2, 3]
            T.toList (T.run T.CPU mask') H.=== [True, False]
        , testProperty "fold preserves nested mixed field order and the initial accumulator" $ H.withTests 1 $ H.property $ do
            let input =
                    ( Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                    , T.fromList @'[] [7 :: Int]
                    )
                result =
                    Structure.foldElements @Show @_ @T.Tensor
                        (\_ acc tensor -> acc ++ map show (T.toList $ T.run T.CPU tensor))
                        ["start"]
                        input
            result H.=== ["start", "2.0", "3.0", "True", "False", "7"]
        , testProperty "fold reads materialized tensor fields without executing a graph" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromListData @'[2] [2, 3]) (T.fromListData @'[2] [True, False])
                result =
                    Structure.foldElements @Show @(Inputs T.Tensor) @T.TensorData
                        (\_ acc tensor -> acc ++ map show (T.toList tensor))
                        []
                        input
            result H.=== ["2.0", "3.0", "True", "False"]
        , testProperty "traverse changes tensor representation without custom field classes" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                result =
                    runIdentity $
                        Structure.traverseElements @T.DType @(Inputs T.Tensor) @T.Tensor @T.TensorData
                            (\_ -> Identity . T.run T.CPU)
                            input
            T.toList (numbers result) H.=== [2, 3]
            T.toList (mask result) H.=== [True, False]
        , testProperty "traverse sequences state effects in field order and preserves values" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                action =
                    Structure.traverseElements @Show @(Inputs T.Tensor) @T.Tensor @T.Tensor
                        ( \_ tensor -> do
                            modify (++ map show (T.toList $ T.run T.CPU tensor))
                            pure tensor
                        )
                        input
                (result, visited) = runState action []
            visited H.=== ["2.0", "3.0", "True", "False"]
            T.toList (T.run T.CPU $ numbers result) H.=== [2, 3]
            T.toList (T.run T.CPU $ mask result) H.=== [True, False]
        , testProperty "traverse propagates a failed field validation" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                result =
                    Structure.traverseElements @Show @(Inputs T.Tensor) @T.Tensor @T.Tensor
                        ( \_ tensor -> if "False" `elem` map show (T.toList $ T.run T.CPU tensor) then Left "false label" else Right tensor
                        )
                        input
            case result of
                Left message -> message H.=== "false label"
                Right _ -> H.failure
        , testProperty "zip combines Bool tensors without differentiability" $ H.withTests 1 $ H.property $ do
            let left = T.fromList @'[2] [True, False]
                right = T.fromList @'[2] [False, True]
                result =
                    Structure.zipWithFields @KeepFirst @(T.Tensor '[2] Bool) (\_ -> keepFirst) left right
            T.toList (T.run T.CPU result) H.=== [True, False]
        , testProperty "general fold receives only the requested field evidence" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromListData @'[2] [2, 3]) (T.fromListData @'[2] [True, False])
                result =
                    Structure.fold @Readable @(Inputs T.Tensor) @T.TensorData
                        (\_ acc tensor -> acc ++ map show (T.toList tensor))
                        []
                        input
            result H.=== ["2.0", "3.0", "True", "False"]
        , testProperty "general traverse accepts a requirement for an input-output representation pair" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                action =
                    Structure.traverse @Materialize @(Inputs T.Tensor) @T.Tensor @T.TensorData
                        (\_ tensor -> modify (+ 1) >> pure (T.run T.CPU tensor))
                        input
                (result, count) = runState action (0 :: Int)
            count H.=== 2
            T.toList (numbers result) H.=== [2, 3]
            T.toList (mask result) H.=== [True, False]
        , testProperty "general zip relates representations shape and dtype in one requirement" $ H.withTests 1 $ H.property $ do
            let left = T.fromListData @'[2] [1, 2 :: Double]
                right = T.fromList @'[2] [3, 4 :: Double]
                result =
                    Structure.zipWith @VectorPair @(T.Tensor '[2] Double) @T.TensorData @T.Tensor @T.Tensor
                        (\path x y -> if null path then T.fromData x + y else error "unexpected root path")
                        left
                        right
            T.toList (T.run T.CPU result) H.=== [4, 6]
        , testProperty "field specializations accept standard classes on mixed dtypes" $ H.withTests 1 $ H.property $ do
            let input = (T.fromList @'[2] [1, 2 :: Float], T.full @'[] (3 :: Double))
                changed = Structure.mapFields @Num (\_ -> (+ 1)) input
                visited = Structure.foldFields @Show (\_ acc tensor -> acc ++ [show tensor]) [] input
                action = Structure.traverseFields @Num (\_ tensor -> modify (+ 1) >> pure (negate tensor)) input
                ((x, y), count) = runState action (0 :: Int)
            length visited H.=== 2
            T.toList (T.run T.CPU $ fst changed) H.=== [2, 3]
            T.toList (T.run T.CPU $ snd changed) H.=== [4]
            count H.=== 2
            T.toList (T.run T.CPU x) H.=== [-1, -2]
            T.toList (T.run T.CPU y) H.=== [-3]
        , testProperty "element map changes representations without a requirement class" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                result = Structure.mapElements @T.DType @(Inputs T.Tensor) @T.Tensor @T.TensorData (\_ -> T.run T.CPU) input
            T.toList (numbers result) H.=== [2, 3]
            T.toList (mask result) H.=== [True, False]
        , testProperty "element zip accepts dtype equality across different tensor shapes" $ H.withTests 1 $ H.property $ do
            let left = (T.fromList @'[2] [1, 2 :: Double], T.full @'[] (3 :: Double))
                right = (T.fromList @'[2] [4, 5 :: Double], T.full @'[] (6 :: Double))
                (x, y) = Structure.zipWithElements @((~) Double) @_ @T.Tensor @T.Tensor @T.Tensor (\_ -> (+)) left right
            T.toList (T.run T.CPU x) H.=== [5, 7]
            T.toList (T.run T.CPU y) H.=== [9]
        , testProperty "materialization preserves records with Bool and numeric tensors" $ H.withTests 1 $ H.property $ do
            let values :: Inputs T.Tensor
                values = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                result = T.run T.CPU values
            T.toList (numbers result) H.=== [2, 3]
            T.toList (mask result) H.=== [True, False]
        , testProperty "prepared records accept changing tensors and masks" $ H.withTests 1 $ H.property $ do
            let function :: Inputs T.Tensor -> T.Tensor '[2] Double
                function p = T.select (mask p) (numbers p) 0
                execute = T.run1 T.CPU function
                first = T.run T.CPU (Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False]))
                second = T.run T.CPU (Inputs (T.fromList @'[2] [5, 7]) (T.fromList @'[2] [False, True]))
            T.toList (execute first) H.=== [2, 0]
            T.toList (execute second) H.=== [0, 7]
        , testProperty "one prepared graph computes gradients and updates nested parameters" $ H.withTests 1 $ H.property $ do
            let initial :: Nested T.Tensor
                initial = Nested (NN.Dense 1 0)
                parameters = T.run T.CPU initial
                execute = T.run2 T.CPU step
                zero = T.run T.CPU (T.full @'[] (0 :: Double))
                rate = T.run T.CPU (T.full @'[] (0.1 :: Double))
                (initialLoss, unchanged) = execute parameters zero
                (_, first) = execute parameters rate
                (nextLoss, second) = execute first rate
            T.toList initialLoss H.=== [4]
            coefficients unchanged H.=== [1, 0]
            approxList (coefficients first) [0.2, -0.4]
            approxList (T.toList nextLoss) [0]
            approxList (coefficients second) [0.2, -0.4]
        ]
