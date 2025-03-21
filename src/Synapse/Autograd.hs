{- | This module implements reverse-mode automatic differentiation.

Machine learning and training of models are based on calculating gradients of operations.
This can be done symbolically by dynamically creating a graph of all operations,
which is then traversed to obtain the gradient.

"Synapse" provides several operations that support automatic differentiation,
but you could easily extend list of those: you just need to define function
that returns 'Symbol' with correct local gradients.
You can check out implementations in the source to give yourself a reference
and read more about it in 'Symbol' datatype docs.
-}


-- 'FlexibleInstances' and 'TypeFamilies' are needed to instantiate 'Indexable', 'ElementwiseScalarOps', 'SingletonOps', 'VecOps', 'MatOps' typeclasses.
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies      #-}


module Synapse.Autograd
    ( -- * 'Symbolic' and 'Symbol'

      Symbolic (symbolicZero, symbolicOne, symbolicN)

    , SymbolIdentifier (SymbolIdentifier, unSymbolIdentifier)

    , Symbol (Symbol, symbolIdentifier, unSymbol, symbolGradients)
    , SymbolVec
    , SymbolMat

    , symbol
    , constSymbol
    , renameSymbol

    , symbolicUnaryOp
    , symbolicBinaryOp

      -- * 'Gradients' calculation

    , Gradients (unGradients)
    , allGradients
    , getGradientsOf
    , wrt
    , nthPartialGradient
    , nthGradient
    ) where


import Synapse.Tensors (DType, Indexable(..), ElementwiseScalarOps(..), ElementwiseScalarOps(..), SingletonOps(..), VecOps(..), MatOps(..))

import Synapse.Tensors.Vec (Vec)
import qualified Synapse.Tensors.Vec as V

import Synapse.Tensors.Mat (Mat)
import qualified Synapse.Tensors.Mat as M

import Data.Foldable (foldl')
import Data.String (IsString(..))

import Data.Hashable (Hashable(..))

import qualified Data.HashMap.Lazy as HM


{- | 'Symbolic' typeclass describes types with few properties that are needed for autogradient.

Members of this typeclass could have default implementation due to 'Num', but such implementation is not always correct.
'Synapse.Tensors.Vec.Vec's and 'Synapse.Tensors.Mat.Mat's do not have only one zero or identity element, and so numerical literal is not enough.
'symbolicZero' and 'symbolicOne' function additionally take reference value to consider dimensions.
Absence of default implementations forces to manually ensure correctness of those functions.

"Synapse" provides implementations for primitive types ('Int', 'Float', 'Double'),
and for containers types ('Synapse.Tensors.Vec.Vec', 'Synapse.Tensors.Vec.Vec').

Detailed laws of 'Symbolic' properties are in the docs for associated functions.
-}
class (Eq a, Num a) => Symbolic a where
    -- | Returns additive and multiplicative (elementwise) zero element. Argument is passed for the reference of the dimension.
    symbolicZero :: a -> a

    -- | Returns multiplicative (elementwise) identity element. Argument is passed for the reference of the dimension.
    symbolicOne :: a -> a

    -- | Returns what could be considered @N@ constant (sum of @N@ 'symbolicOne's). Argument is passed for the reference of the dimension.
    symbolicN :: Int -> a -> a
    symbolicN n c
        | n < 0     = negate $ go (abs n) c
        | n == 0    = symbolicZero c
        | otherwise = go n c
      where
        go 0 x = x
        go i x = go (i - 1) (x + symbolicOne x)

-- Instances

instance Symbolic Int where
    symbolicZero = const 0
    symbolicOne = const 1
    symbolicN = const

instance Symbolic Float where
    symbolicZero = const 0.0
    symbolicOne = const 1.0
    symbolicN n = const (fromIntegral n)

instance Symbolic Double where
    symbolicZero = const 0.0
    symbolicOne = const 1.0
    symbolicN n = const (fromIntegral n)


instance Symbolic a => Symbolic (Vec a) where
    symbolicZero reference = V.replicate (V.size reference) 0
    symbolicOne reference = V.replicate (V.size reference) 1
    symbolicN n reference = V.replicate (V.size reference) (fromIntegral n)

instance Symbolic a => Symbolic (Mat a) where
    symbolicZero reference = M.replicate (M.size reference) 0
    symbolicOne reference = M.replicate (M.size reference) 1
    symbolicN n reference = M.replicate (M.size reference) (fromIntegral n)


-- | 'SymbolIdentifier' is a newtype that wraps string, which needs to uniquely represent symbol.
newtype SymbolIdentifier = SymbolIdentifier 
    { unSymbolIdentifier :: String  -- ^ Identifier of a symbol.
    } deriving (Eq, Show)

instance IsString SymbolIdentifier where
    fromString = SymbolIdentifier

instance Semigroup SymbolIdentifier where
    (<>) (SymbolIdentifier a) (SymbolIdentifier b) = SymbolIdentifier $ a <> b

instance Monoid SymbolIdentifier where
    mempty = SymbolIdentifier ""

{- | Datatype that represents symbol variable (variable which operations are recorded to symbolically obtain derivatives).

Any operation returning @Symbol a@ where @a@ is 'Symbolic' could be autogradiented - returned 'Symbol' has 'symbolGradients' list,
which allows "Synapse" to build a graph of computation and obtain needed gradients.
'symbolGradients' list contains pairs: first element in that pair is symbol wrt which you can take gradient and
the second element is closure that represents chain rule - it takes incoming local gradient of said symbol and multiplies it by local derivative.
You can check out implementations of those operations in the source to give yourself a reference.
-}
data Symbol a = Symbol
    { symbolIdentifier :: SymbolIdentifier                    -- ^ Name of a symbol (identifier for differentiation).
    , unSymbol         :: a                                   -- ^ Value of a symbol.
    , symbolGradients  :: [(Symbol a, Symbol a -> Symbol a)]  -- ^ List of gradients (wrt to what Symbol and closure to calculate gradient). 
    }

-- | Creates new symbol that refers to a variable (so it must have a name to be able to be differentiated wrt).
symbol :: SymbolIdentifier -> a -> Symbol a
symbol name value = Symbol name value []

-- | Creates new symbol that refers to constant (so it does not have name and thus its gradients are not saved).
constSymbol :: a -> Symbol a
constSymbol = symbol mempty

-- | Renames symbol which allows differentiating wrt it. Note: renaming practically creates new symbol for the gradient calculation.
renameSymbol :: SymbolIdentifier -> Symbol a -> Symbol a
renameSymbol name (Symbol _ value localGradients) = Symbol name value localGradients


-- | @SymbolVec a@ type alias stands for @Symbol (Vec a)@.
type SymbolVec a = Symbol (Vec a)

-- | @SymbolMat a@ type alias stands for @Symbol (Mat a)@.
type SymbolMat a = Symbol (Mat a)


-- Typeclasses

instance Show a => Show (Symbol a) where
    show (Symbol name value _) = "Symbol " ++ show name ++ ": " ++ show value


instance Eq (Symbol a) where
    (==) (Symbol name1 _ _) (Symbol name2 _ _) = name1 == name2


instance Hashable (Symbol a) where
    hashWithSalt salt (Symbol (SymbolIdentifier name) _ _) = hashWithSalt salt name


-- Symbolic symbols

instance Symbolic a => Symbolic (Symbol a) where
    symbolicZero x = constSymbol $ symbolicZero $ unSymbol x
    symbolicOne x = constSymbol $ symbolicOne $ unSymbol x


type instance DType (SymbolVec a) = a

type instance DType (SymbolMat a) = a


-- | Converts unary operation into symbolic one.
symbolicUnaryOp :: (a -> a) -> Symbol a -> [(Symbol a, Symbol a -> Symbol a)] -> Symbol a
symbolicUnaryOp op x = Symbol mempty (op (unSymbol x))

-- | Converts binary operation into symbolic one.
symbolicBinaryOp :: (a -> a -> a) -> Symbol a -> Symbol a -> [(Symbol a, Symbol a -> Symbol a)] -> Symbol a
symbolicBinaryOp op a b = Symbol mempty (op (unSymbol a) (unSymbol b))

instance Symbolic a => Num (Symbol a) where
    (+) a b = symbolicBinaryOp (+) a b [(a, id), (b, id)]
    (-) a b = symbolicBinaryOp (-) a b [(a, id), (b, negate)]
    negate x = symbolicUnaryOp negate x [(x, negate)]
    (*) a b = symbolicBinaryOp (*) a b [(a, (* b)), (b, (a *))]
    abs x = symbolicUnaryOp abs x [(x, signum)]
    signum x = symbolicUnaryOp signum x [(x, id)]
    fromInteger = constSymbol . fromInteger

instance (Symbolic a, Fractional a) => Fractional (Symbol a) where
    (/) a b = symbolicBinaryOp (/) a b [(a, (/ b)), (b, (* (negate a / (b * b))))]
    recip x = symbolicUnaryOp recip x [(x, (* (negate (symbolicOne x) / (x * x))))]
    fromRational = constSymbol . fromRational

instance (Symbolic a, Floating a) => Floating (Symbol a) where
    pi = constSymbol pi
    (**) a b = symbolicBinaryOp (**) a b [(a, (* (b * a ** (b - symbolicOne b)))), (b, (* (a ** b * log a)))]
    sqrt x = symbolicUnaryOp sqrt x [(x, (* (recip $ symbolicN 2 x * sqrt x)))]
    exp x = symbolicUnaryOp exp x [(x, (* exp x))]
    log x = symbolicUnaryOp log x [(x, (* recip x))]
    sin x = symbolicUnaryOp sin x [(x, (* cos x))]
    cos x = symbolicUnaryOp cos x [(x, (* negate (sin x)))]
    asin x = symbolicUnaryOp asin x [(x, (* recip (sqrt (symbolicOne x - x * x))))]
    acos x = symbolicUnaryOp acos x [(x, (* negate (recip (sqrt (symbolicOne x - x * x)))))]
    atan x = symbolicUnaryOp atan x [(x, (* recip (symbolicOne x + x * x)))]
    sinh x = symbolicUnaryOp sinh x [(x, (* cosh x))]
    cosh x = symbolicUnaryOp cosh x [(x, (* sinh x))]
    asinh x = symbolicUnaryOp asinh x [(x, (* recip (sqrt (symbolicOne x + x * x))))]
    acosh x = symbolicUnaryOp acosh x [(x, (* recip (sqrt (x * x - symbolicOne x))))]
    atanh x = symbolicUnaryOp atanh x [(x, (* recip (symbolicOne x - x * x)))]

instance Symbolic a => ElementwiseScalarOps (Symbol (Vec a)) where
    (+.) x n = x + constSymbol (V.replicate (V.size $ unSymbol x) n)
    (-.) x n = x - constSymbol (V.replicate (V.size $ unSymbol x) n)
    (*.) x n = x * constSymbol (V.replicate (V.size $ unSymbol x) n)
    (/.) x n = x / constSymbol (V.replicate (V.size $ unSymbol x) n)
    (**.) x n = x ** constSymbol (V.replicate (V.size $ unSymbol x) n)

    elementsMin x n = symbolicUnaryOp (`elementsMin` n) x
                      [(x, (* constSymbol (V.generate (V.size $ unSymbol x) $ \i -> if unsafeIndex (unSymbol x) i <= n then 1 else 0)))]
    elementsMax x n = symbolicUnaryOp (`elementsMax` n) x
                      [(x, (* constSymbol (V.generate (V.size $ unSymbol x) $ \i -> if unsafeIndex (unSymbol x) i >= n then 1 else 0)))]

instance Symbolic a => ElementwiseScalarOps (SymbolMat a) where
    (+.) x n = x + constSymbol (M.replicate (M.size $ unSymbol x) n)
    (-.) x n = x - constSymbol (M.replicate (M.size $ unSymbol x) n)
    (*.) x n = x * constSymbol (M.replicate (M.size $ unSymbol x) n)
    (/.) x n = x / constSymbol (M.replicate (M.size $ unSymbol x) n)
    (**.) x n = x ** constSymbol (M.replicate (M.size $ unSymbol x) n)

    elementsMin x n = symbolicUnaryOp (`elementsMin` n) x
                      [(x, (* constSymbol (M.generate (M.size $ unSymbol x) $ \i -> if unsafeIndex (unSymbol x) i <= n then 1 else 0)))]
    elementsMax x n = symbolicUnaryOp (`elementsMax` n) x
                      [(x, (* constSymbol (M.generate (M.size $ unSymbol x) $ \i -> if unsafeIndex (unSymbol x) i >= n then 1 else 0)))]

instance Symbolic a => SingletonOps (SymbolVec a) where
    singleton = constSymbol . singleton
    isSingleton = isSingleton . unSymbol
    unSingleton = unSingleton . unSymbol

    extendSingleton vec reference = constSymbol $ extendSingleton (unSymbol vec) (unSymbol reference)

    elementsSum x = symbolicUnaryOp elementsSum x [(x, (`extendSingleton` x))]
    elementsProduct x = let innerProduct = unSingleton $ elementsProduct $ unSymbol x
                        in symbolicUnaryOp elementsProduct x [(x, \path -> extendSingleton path x * constSymbol (V.generate (V.size $ unSymbol x) $ \i -> innerProduct / unsafeIndex (unSymbol x) i))]

    mean x = symbolicUnaryOp mean x [(x, \path -> extendSingleton path x /. fromIntegral (V.size (unSymbol x)))]

    norm x = symbolicUnaryOp norm x [(x, \path -> extendSingleton path x * (x /. unSingleton (norm $ unSymbol x)))]

instance Symbolic a => SingletonOps (SymbolMat a) where
    singleton = constSymbol . singleton
    isSingleton = isSingleton . unSymbol
    unSingleton = unSingleton . unSymbol

    extendSingleton mat reference = constSymbol $ extendSingleton (unSymbol mat) (unSymbol reference)

    elementsSum x = symbolicUnaryOp elementsSum x [(x, (`extendSingleton` x))]
    elementsProduct x = let innerProduct = unSingleton $ elementsProduct $ unSymbol x
                        in symbolicUnaryOp elementsProduct x [(x, \path -> extendSingleton path x * constSymbol (M.generate (M.size $ unSymbol x) $ \i -> innerProduct / unsafeIndex (unSymbol x) i))]

    mean x = symbolicUnaryOp mean x [(x, \path -> extendSingleton path x /. fromIntegral (M.nElements $ unSymbol x))]

    norm x = symbolicUnaryOp norm x [(x, \path -> extendSingleton path x * (x /. unSingleton (norm $ unSymbol x)))]

instance Symbolic a => VecOps (SymbolVec a) where
    dot a b = elementsSum $ a * b

instance Symbolic a => MatOps (SymbolMat a) where
    transpose x = symbolicUnaryOp M.transpose x [(x, (* transpose x))]

    addMatRow mat row = symbolicBinaryOp addMatRow mat row [(mat, id), (row, constSymbol . M.rowVec . flip M.indexRow 0 . unSymbol)]

    matMul a b = symbolicBinaryOp M.matMul a b [(a, (`matMul` transpose b)), (b, (transpose a `matMul`))]


-- Gradients calculation

-- | 'Gradients' datatype holds all gradients of one symbol with respect to other symbols.
newtype Gradients a = Gradients
    { unGradients :: HM.HashMap (Symbol a) (Symbol a)  -- ^ Map of gradients.
    }

-- | Returns key-value pairs of all gradients of symbol.
allGradients :: Gradients a -> [(Symbol a, Symbol a)]
allGradients = HM.toList . unGradients


-- Typeclasses

instance Show a => Show (Gradients a) where
    show gradients = show $ allGradients gradients


-- | Generates 'Gradients' for given symbol.
getGradientsOf :: Symbolic a => Symbol a -> Gradients a
getGradientsOf differentiatedSymbol = Gradients $ HM.insert differentiatedSymbol wrtItself $
                                                  HM.delete (Symbol mempty undefined []) $
                                                  go HM.empty differentiatedSymbol wrtItself
  where
    wrtItself = symbolicOne differentiatedSymbol

    go grads s pathValue =
        foldr (\(child, mulPath) grad -> let pathValue' = mulPath pathValue
                                             grad' = HM.alter (\e -> Just $ case e of
                                                                                Nothing -> pathValue'
                                                                                Just x  -> x + pathValue'
                                                              ) child grad
                                         in go grad' child pathValue') grads (symbolGradients s)

-- | Chooses gradient with respect to given symbol.
wrt :: Symbolic a => Gradients a -> Symbol a -> Symbol a
wrt gradients x = HM.findWithDefault (symbolicZero x) x $ unGradients gradients

-- | Takes partial gradients wrt to all symbols in a list sequentially, returning last result.
nthPartialGradient :: Symbolic a => Symbol a -> [Symbol a] -> Symbol a
nthPartialGradient = foldl' $ \y x -> getGradientsOf y `wrt` x

-- | Takes nth order gradient of one symbol wrt other symbol. If n is negative number, an error is returned.
nthGradient :: Symbolic a => Int -> Symbol a -> Symbol a -> Symbol a
nthGradient n y x
    | n < 0 = error "Cannot take negative order gradient"
    | otherwise = nthPartialGradient y (replicate n x)
