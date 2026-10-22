{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module AutogradTest.Differentiable (tests) where

import GHC.TypeLits (Nat)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Autograd.Differentiable as D
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

newtype Single a = Single {singleValue :: a} deriving (Eq, Show)

instance (AG.Differentiable a) => AG.Differentiable (Single a) where
  type WithGrad s (Single a) = Single (AG.WithGrad s a)
  structure =
    D.record Single Single
      D.<+> D.field singleValue singleValue

data Position = Position
  { positionX :: Double,
    positionY :: Float,
    positionZ :: Double
  }
  deriving (Eq, Show)

data PositionVar s = PositionVar
  { variableX :: AG.Var s Double,
    variableY :: AG.Var s Float,
    variableZ :: AG.Var s Double
  }

instance AG.Differentiable Position where
  type WithGrad s Position = PositionVar s
  structure =
    D.record Position PositionVar
      D.<+> D.field positionX variableX
      D.<+> D.field positionY variableY
      D.<+> D.field positionZ variableZ

data Wide a = Wide
  { first :: a,
    second :: a,
    third :: a,
    fourth :: a,
    fifth :: a,
    sixth :: a,
    seventh :: a,
    eighth :: a,
    ninth :: a,
    tenth :: a,
    eleventh :: a,
    twelfth :: a
  }
  deriving (Eq, Show)

instance (AG.Differentiable a) => AG.Differentiable (Wide a) where
  type WithGrad s (Wide a) = Wide (AG.WithGrad s a)
  structure =
    D.record Wide Wide
      D.<+> D.field first first
      D.<+> D.field second second
      D.<+> D.field third third
      D.<+> D.field fourth fourth
      D.<+> D.field fifth fifth
      D.<+> D.field sixth sixth
      D.<+> D.field seventh seventh
      D.<+> D.field eighth eighth
      D.<+> D.field ninth ninth
      D.<+> D.field tenth tenth
      D.<+> D.field eleventh eleventh
      D.<+> D.field twelfth twelfth

data Parameters t (input :: Nat) (output :: Nat) a = Parameters
  { weights :: t '[input, output] a,
    bias :: t '[output] a,
    scale :: t '[] a
  }

instance
  ( AG.Differentiable (T.Tensor '[input, output] a),
    AG.Differentiable (T.Tensor '[output] a),
    AG.Differentiable (T.Tensor '[] a)
  ) =>
  AG.Differentiable (Parameters T.Tensor input output a)
  where
  type WithGrad s (Parameters T.Tensor input output a) = Parameters (AG.Tensor s) input output a
  structure =
    D.record Parameters Parameters
      D.<+> D.field weights weights
      D.<+> D.field bias bias
      D.<+> D.field scale scale

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "differentiable records"
    [ testProperty "manual record constants preserve their custom representation" $ H.withTests 1 $ H.property $ do
        let frozen = Position 3 4 5
            objective x = let PositionVar a b c = AG.constant frozen in PositionVar (x * a) b c
        AG.backprop objective (2 :: Double) H.=== (Position 6 4 5, 3)
        AG.evaluate (const (AG.constant frozen)) (0 :: Double) H.=== frozen,
      testProperty "a single field completes the record without a finishing operation" $ H.withTests 1 $ H.property $ do
        AG.backprop (\(Single x) -> Single (x * x)) (Single (3 :: Double)) H.=== (Single 9, Single 6)
        AG.evaluate (\(Single x) -> Single (x * x)) (Single (3 :: Double)) H.=== Single 9,
      testProperty "nested manual records register their leaves independently" $ H.withTests 1 $ H.property $ do
        let input = Single (Single (3 :: Double))
        AG.backprop (\(Single (Single x)) -> Single (Single (x * x))) input H.=== (Single (Single 9), Single (Single 6)),
      testProperty "ordinary and differentiable records can have different constructors and selectors" $ H.withTests 1 $ H.property $ do
        let objective (PositionVar x y z) = (x * x, y * y, z * x)
            input = Position 3 4 5
            (value, backward) = AG.pullback objective input
        value H.=== (9, 16, 15)
        backward (2, 3, 4) H.=== Position 32 24 12
        backward (1, 1, 1) H.=== Position 11 8 3
        AG.evaluate objective input H.=== (9, 16, 15)
        AG.gradient (objective . AG.stopGrad) input H.=== Position 0 0 0,
      testProperty "manual records support more than nine fields" $ H.withTests 1 $ H.property $ do
        let input = Wide 1 2 3 4 5 6 7 8 9 10 11 12 :: Wide Double
            (value, gradient) = AG.backprop id input
        value H.=== input
        gradient H.=== Wide 1 1 1 1 1 1 1 1 1 1 1 1
        AG.gradient (\record -> first record + twelfth record) input H.=== Wide 1 0 0 0 0 0 0 0 0 0 0 1,
      testProperty "shape and element parameters remain inferable in a three-field tensor record" $ H.withTests 1 $ H.property $ do
        let input = Parameters (T.unsafeFromList [1 .. 6]) (T.unsafeFromList [7, 8, 9]) (T.full 2) :: Parameters T.Tensor 2 3 Double
            objective (Parameters w b multiplier) = T.sum (w * w + T.broadcast b) * multiplier
            (value, Parameters dw db ds) = AG.backprop objective input
            Parameters unusedW unusedB unusedS = AG.gradient (\(Parameters w _ _) -> T.sum w) input
        values value H.=== [278]
        values dw H.=== [4, 8, 12, 16, 20, 24]
        values db H.=== [4, 4, 4]
        values ds H.=== [139]
        values unusedW H.=== replicate 6 1
        values unusedB H.=== replicate 3 0
        values unusedS H.=== [0],
      testProperty "a record cannot omit a constructor argument" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "MissingField.hs") "Couldn't match",
      testProperty "a record cannot add fields after the constructor is complete" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "ExtraField.hs") "Couldn't match",
      testProperty "a field must match its constructor argument type" $ H.withTests 1 $ Compiler.rejects (fixtures ++ "FieldType.hs") "Couldn't match"
    ]
  where
    fixtures = "test/AutogradTest/Differentiable/Fixtures/Rejected/"
