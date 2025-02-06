{- | This module implements reverse-mode automatic differentiation.

Machine learning and training of models are based on calculating gradients of operations.
This can be done symbolically by dynamically creating a graph of all operations,
which is then traversed to obtain the gradient.

@Synapse@ provides several operations that support automatic differentiation,
but you could easily extend list of those: you just need to define function
that takes @Symbol@(s) and returns @Symbol@ with correct local gradients.
You can check out implementations in the source to give yourself a reference.
-}


module Synapse.Autograd
    ( -- * @Symbol@ and @Symbolic@

      Symbol (Symbol, symbolName, unSymbol, symbolGradients)
    , Symbolic (symbolZero, symbolOne)

    , symbol
    , constSymbol
    , renameSymbol

      -- * @Gradients@ calculation

    , Gradients (unGradients)
    , getGradientsOf
    , wrt
    , toList
    ) where


import Data.Hashable (Hashable(..))

import qualified Data.HashMap.Lazy as HM


-- | Datatype that represents symbol variable (variable which operations are recorded to symbolically obtain derivatives).
data Symbol a = Symbol
    { symbolName      :: String                              -- ^ Name of a symbol (identifier for differentiation).
    , unSymbol        :: a                                   -- ^ Value of a symbol.
    , symbolGradients :: [(Symbol a, Symbol a -> Symbol a)]  -- ^ List of gradients (wrt to what Symbol and closure to calculate gradient). 
    }


-- | Creates new symbol that refers to a variable (so it must have a name to be able to be differentiated wrt).
symbol :: String -> a -> Symbol a
symbol name value = Symbol name value []

-- | Creates new symbol that refers to constant (so it does not have name).
constSymbol :: a -> Symbol a
constSymbol = symbol ""

-- | Renames symbol which allows differentiating wrt it. Note: renaming practically creates new symbol for the gradient calculation.
renameSymbol :: String -> Symbol a -> Symbol a
renameSymbol name (Symbol _ value localGradients) = Symbol name value localGradients


{- | @Symbolic@ typeclass describes few properties that are needed for autogradient.

Members of this typeclass could have default implementation due to @Num@, but such implementation is not always correct.
@Vec@s and @Mat@s do not have only one zero or identity element, and so numerical literal is not enough.
@symbolZero@ and @symbolOne@ function additionally take reference value to consider dimensions.
Correct implementation of those functions is important.

Detailed laws of @Symbolic@ properties are in the docs for associated functions.
-}
class Num a => Symbolic a where
    -- | Returns what could be considered as additive and multiplicative zero element. Argument is passed for the reference of the dimension.
    symbolZero :: a -> a

    -- | Returns what could be considered as multiplicative identity element. Argument is passed for the reference of the dimension.
    symbolOne :: a -> a


-- Typeclasses

instance Show a => Show (Symbol a) where
    show (Symbol name value _) = "Symbol " ++ show name ++ ": " ++ show value


instance Num a => Num (Symbol a) where
    (+) a b = Symbol "" (unSymbol a + unSymbol b) [(a, id), (b, id)]
    (-) a b = Symbol "" (unSymbol a - unSymbol b) [(a, id), (b, negate)]
    negate x = Symbol "" (negate $ unSymbol x) [(x, negate)]
    (*) a b = Symbol "" (unSymbol a * unSymbol b) [(a, (* b)), (b, (a *))]
    abs x = Symbol "" (abs $ unSymbol x) [(x, signum)]
    signum x = Symbol "" (signum $ unSymbol x) [(x, id)]
    fromInteger x = constSymbol $ fromInteger x
    


instance Eq (Symbol a) where
    (==) (Symbol name1 _ _) (Symbol name2 _ _) = name1 == name2

instance Hashable (Symbol a) where
    hashWithSalt salt (Symbol name _ _) = hashWithSalt salt name


-- Gradients calculation

-- | Datatype that holds all gradients of one symbol with respect to other symbols.
newtype Gradients a = Gradients 
    { unGradients :: HM.HashMap (Symbol a) (Symbol a)  -- ^ Map of gradients.
    } deriving Show


-- | Generates @Gradients@ for given symbol.
getGradientsOf :: Symbolic a => Symbol a -> Gradients a
getGradientsOf differentiatedSymbol = Gradients $ go HM.empty differentiatedSymbol (constSymbol $ symbolOne $ unSymbol differentiatedSymbol)
  where
    go :: Symbolic a => HM.HashMap (Symbol a) (Symbol a) -> Symbol a -> Symbol a -> HM.HashMap (Symbol a) (Symbol a)
    go grads (Symbol _ _ localGrads) pathValue =
        foldr (\(child, mulPath) grad -> let pathValue' = mulPath pathValue
                                             grad' = HM.insertWith (\_ old -> old + pathValue') 
                                                                   child (constSymbol $ symbolZero $ unSymbol child) grad
                                         in go grad' child pathValue') grads localGrads

-- | Chooses gradient with respect to given symbol.
wrt :: Gradients a -> Symbol a -> Maybe (Symbol a)
wrt gradients x = HM.lookup x $ unGradients gradients

-- | Returns key-value pairs of all gradients of symbol.
toList :: Gradients a -> [(Symbol a, Symbol a)]
toList = HM.toList . unGradients

