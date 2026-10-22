{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Shape-preserving elementwise tensor operations.
module Synapse.Tensor.Ops.Elementwise
  ( ElementControl (..),
    Elementwise (..),
    Boolean (..),
    Conditional (..),
    Eq (..),
    Ord (..),
    Integral (..),
    RealFloat (..),
    map,
    zip,
    zip3,
    zip4,
    zip5,
    zip6,
    zip7,
    zip8,
    zip9,
    zipWith,
    zipWith3,
    zipWith4,
    zipWith5,
    zipWith6,
    zipWith7,
    zipWith8,
    zipWith9,
    unzip,
    unzip3,
    unzip4,
    unzip5,
    unzip6,
    unzip7,
    unzip8,
    unzip9,
    while,
    iterate,
    pair,
    fst,
    snd,
  )
where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Element, Tensor (..), TensorElement (..))
import Prelude hiding (Eq, Integral, Ord, RealFloat, fst, isInfinite, isNaN, iterate, map, max, min, not, quot, rem, snd, unzip, unzip3, zip, zip3, zipWith, zipWith3, (&&), (/=), (<), (<=), (==), (>), (>=), (||))

-- | Tuple expressions and element-level control flow.
class ElementControl element where
  -- | Combine two element expressions into a pair, including loop states.
  pairOp ::
    forall a b.
    (DType a, DType b) =>
    element a ->
    element b ->
    element (a, b)

  -- | Extract the first expression from an element pair.
  fstOp ::
    forall a b.
    (DType a, DType b) =>
    element (a, b) ->
    element a

  -- | Extract the second expression from an element pair.
  sndOp ::
    forall a b.
    (DType a, DType b) =>
    element (a, b) ->
    element b

  -- | Apply a state update while the predicate holds, testing it before every update.
  -- The state type remains unchanged and the caller must ensure termination.
  whileOp ::
    forall a.
    (DType a) =>
    (element a -> element Bool) ->
    (element a -> element a) ->
    element a ->
    element a

  -- | Apply an element state update a specified number of times.
  -- The iteration count is an element expression, so it can depend on the current tensor index.
  iterateOp ::
    forall a.
    (DType a) =>
    element Int ->
    (element a -> element a) ->
    element a ->
    element a

instance ElementControl TensorElement where
  pairOp (TensorElement x) (TensorElement y) = TensorElement (A.lift (x, y))

  fstOp (TensorElement x) = TensorElement (A.fst x)

  sndOp (TensorElement x) = TensorElement (A.snd x)

  whileOp predicate step (TensorElement initial) =
    TensorElement (A.while (unTensorElement . predicate . TensorElement) (unTensorElement . step . TensorElement) initial)

  iterateOp (TensorElement count) step (TensorElement initial) =
    TensorElement (A.iterate count (unTensorElement . step . TensorElement) initial)

-- | Combine two element expressions into a pair, including loop states.
pair ::
  forall {element} a b.
  (ElementControl element, DType a, DType b) =>
  element a ->
  element b ->
  element (a, b)
pair = pairOp @element @a @b

-- | Extract the first expression from an element pair.
fst ::
  forall {element} a b.
  (ElementControl element, DType a, DType b) =>
  element (a, b) ->
  element a
fst = fstOp @element @a @b

-- | Extract the second expression from an element pair.
snd ::
  forall {element} a b.
  (ElementControl element, DType a, DType b) =>
  element (a, b) ->
  element b
snd = sndOp @element @a @b

-- | Apply a state update while the predicate holds, testing it before every update.
-- The state type remains unchanged and the caller must ensure termination.
while ::
  forall {element} a.
  (ElementControl element, DType a) =>
  (element a -> element Bool) ->
  (element a -> element a) ->
  element a ->
  element a
while = whileOp @element @a

-- | Apply an element state update a specified number of times.
-- The iteration count is an element expression, so it can depend on the current tensor index.
iterate ::
  forall {element} a.
  (ElementControl element, DType a) =>
  element Int ->
  (element a -> element a) ->
  element a ->
  element a
iterate = iterateOp @element @a

-- | Shape-preserving transformations and tuple operations.
class Elementwise t where
  -- | Transform each element, preserving the shape while allowing a new element type.
  mapOp ::
    forall sh a b.
    (Shape sh, DType a, DType b) =>
    (Element t a -> Element t b) ->
    t sh a ->
    t sh b

  -- | Pair corresponding elements of two tensors with the same shape.
  zipOp ::
    forall sh a b.
    (Shape sh, DType a, DType b) =>
    t sh a ->
    t sh b ->
    t sh (a, b)

  -- | Group corresponding elements of three equally shaped tensors into triples.
  -- The three element types may differ.
  zip3Op ::
    forall sh a b c.
    (Shape sh, DType a, DType b, DType c) =>
    t sh a ->
    t sh b ->
    t sh c ->
    t sh (a, b, c)

  -- | Group corresponding elements of 4 equally shaped tensors into tuples.
  -- Input element types may differ.
  zip4Op ::
    forall sh a b c d.
    (Shape sh, DType a, DType b, DType c, DType d) =>
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh (a, b, c, d)

  -- | Group corresponding elements of 5 equally shaped tensors into tuples.
  -- Input element types may differ.
  zip5Op ::
    forall sh a b c d e.
    (Shape sh, DType a, DType b, DType c, DType d, DType e) =>
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh (a, b, c, d, e)

  -- | Group corresponding elements of 6 equally shaped tensors into tuples.
  -- Input element types may differ.
  zip6Op ::
    forall sh a b c d e f.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f) =>
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f ->
    t sh (a, b, c, d, e, f)

  -- | Group corresponding elements of 7 equally shaped tensors into tuples.
  -- Input element types may differ.
  zip7Op ::
    forall sh a b c d e f g.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g) =>
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f ->
    t sh g ->
    t sh (a, b, c, d, e, f, g)

  -- | Group corresponding elements of 8 equally shaped tensors into tuples.
  -- Input element types may differ.
  zip8Op ::
    forall sh a b c d e f g h.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h) =>
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f ->
    t sh g ->
    t sh h ->
    t sh (a, b, c, d, e, f, g, h)

  -- | Group corresponding elements of 9 equally shaped tensors into tuples.
  -- Input element types may differ.
  zip9Op ::
    forall sh a b c d e f g h i.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i) =>
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f ->
    t sh g ->
    t sh h ->
    t sh i ->
    t sh (a, b, c, d, e, f, g, h, i)

  -- | Combine corresponding elements of two tensors with the same shape.
  zipWithOp ::
    forall sh a b c.
    (Shape sh, DType a, DType b, DType c) =>
    (Element t a -> Element t b -> Element t c) ->
    t sh a ->
    t sh b ->
    t sh c

  -- | Combine corresponding elements of three equally shaped tensors.
  -- All input element types and the output element type may differ.
  zipWith3Op ::
    forall sh a b c d.
    (Shape sh, DType a, DType b, DType c, DType d) =>
    (Element t a -> Element t b -> Element t c -> Element t d) ->
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d

  -- | Combine corresponding elements of 4 equally shaped tensors.
  -- All input element types and the output element type may differ.
  zipWith4Op ::
    forall sh a b c d e.
    (Shape sh, DType a, DType b, DType c, DType d, DType e) =>
    (Element t a -> Element t b -> Element t c -> Element t d -> Element t e) ->
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e

  -- | Combine corresponding elements of 5 equally shaped tensors.
  -- All input element types and the output element type may differ.
  zipWith5Op ::
    forall sh a b c d e f.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f) =>
    (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f) ->
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f

  -- | Combine corresponding elements of 6 equally shaped tensors.
  -- All input element types and the output element type may differ.
  zipWith6Op ::
    forall sh a b c d e f g.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g) =>
    (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f -> Element t g) ->
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f ->
    t sh g

  -- | Combine corresponding elements of 7 equally shaped tensors.
  -- All input element types and the output element type may differ.
  zipWith7Op ::
    forall sh a b c d e f g h.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h) =>
    (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f -> Element t g -> Element t h) ->
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f ->
    t sh g ->
    t sh h

  -- | Combine corresponding elements of 8 equally shaped tensors.
  -- All input element types and the output element type may differ.
  zipWith8Op ::
    forall sh a b c d e f g h i.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i) =>
    (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f -> Element t g -> Element t h -> Element t i) ->
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f ->
    t sh g ->
    t sh h ->
    t sh i

  -- | Combine corresponding elements of 9 equally shaped tensors.
  -- All input element types and the output element type may differ.
  zipWith9Op ::
    forall sh a b c d e f g h i j.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i, DType j) =>
    (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f -> Element t g -> Element t h -> Element t i -> Element t j) ->
    t sh a ->
    t sh b ->
    t sh c ->
    t sh d ->
    t sh e ->
    t sh f ->
    t sh g ->
    t sh h ->
    t sh i ->
    t sh j

  -- | Split a tensor of 2-tuples into 2 tensor expressions with the same shape.
  -- Component element types may differ and no computation is executed on the host.
  unzipOp ::
    forall sh a b.
    (Shape sh, DType a, DType b) =>
    t sh (a, b) ->
    (t sh a, t sh b)

  -- | Split a tensor of 3-tuples into 3 tensor expressions with the same shape.
  -- Component element types may differ and no computation is executed on the host.
  unzip3Op ::
    forall sh a b c.
    (Shape sh, DType a, DType b, DType c) =>
    t sh (a, b, c) ->
    (t sh a, t sh b, t sh c)

  -- | Split a tensor of 4-tuples into 4 tensor expressions with the same shape.
  -- Component element types may differ and no computation is executed on the host.
  unzip4Op ::
    forall sh a b c d.
    (Shape sh, DType a, DType b, DType c, DType d) =>
    t sh (a, b, c, d) ->
    (t sh a, t sh b, t sh c, t sh d)

  -- | Split a tensor of 5-tuples into 5 tensor expressions with the same shape.
  -- Component element types may differ and no computation is executed on the host.
  unzip5Op ::
    forall sh a b c d e.
    (Shape sh, DType a, DType b, DType c, DType d, DType e) =>
    t sh (a, b, c, d, e) ->
    (t sh a, t sh b, t sh c, t sh d, t sh e)

  -- | Split a tensor of 6-tuples into 6 tensor expressions with the same shape.
  -- Component element types may differ and no computation is executed on the host.
  unzip6Op ::
    forall sh a b c d e f.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f) =>
    t sh (a, b, c, d, e, f) ->
    (t sh a, t sh b, t sh c, t sh d, t sh e, t sh f)

  -- | Split a tensor of 7-tuples into 7 tensor expressions with the same shape.
  -- Component element types may differ and no computation is executed on the host.
  unzip7Op ::
    forall sh a b c d e f g.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g) =>
    t sh (a, b, c, d, e, f, g) ->
    (t sh a, t sh b, t sh c, t sh d, t sh e, t sh f, t sh g)

  -- | Split a tensor of 8-tuples into 8 tensor expressions with the same shape.
  -- Component element types may differ and no computation is executed on the host.
  unzip8Op ::
    forall sh a b c d e f g h.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h) =>
    t sh (a, b, c, d, e, f, g, h) ->
    (t sh a, t sh b, t sh c, t sh d, t sh e, t sh f, t sh g, t sh h)

  -- | Split a tensor of 9-tuples into 9 tensor expressions with the same shape.
  -- Component element types may differ and no computation is executed on the host.
  unzip9Op ::
    forall sh a b c d e f g h i.
    (Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i) =>
    t sh (a, b, c, d, e, f, g, h, i) ->
    (t sh a, t sh b, t sh c, t sh d, t sh e, t sh f, t sh g, t sh h, t sh i)

instance Elementwise Tensor where
  mapOp f (Tensor x) = Tensor (A.map (unTensorElement . f . TensorElement) x)

  zipOp (Tensor x) (Tensor y) = Tensor (A.zip x y)

  zip3Op (Tensor x) (Tensor y) (Tensor z) = Tensor (A.zip3 x y z)

  zip4Op (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) = Tensor (A.zip4 x1 x2 x3 x4)

  zip5Op (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) = Tensor (A.zip5 x1 x2 x3 x4 x5)

  zip6Op (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) = Tensor (A.zip6 x1 x2 x3 x4 x5 x6)

  zip7Op (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7) = Tensor (A.zip7 x1 x2 x3 x4 x5 x6 x7)

  zip8Op (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7) (Tensor x8) = Tensor (A.zip8 x1 x2 x3 x4 x5 x6 x7 x8)

  zip9Op (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7) (Tensor x8) (Tensor x9) = Tensor (A.zip9 x1 x2 x3 x4 x5 x6 x7 x8 x9)

  zipWithOp f (Tensor x) (Tensor y) = Tensor $ A.zipWith (\a b -> unTensorElement (f (TensorElement a) (TensorElement b))) x y

  zipWith3Op f (Tensor x) (Tensor y) (Tensor z) =
    Tensor (A.zipWith3 (\a b c -> unTensorElement (f (TensorElement a) (TensorElement b) (TensorElement c))) x y z)

  zipWith4Op f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) =
    Tensor (A.zipWith4 (\a1 a2 a3 a4 -> unTensorElement (f (TensorElement a1) (TensorElement a2) (TensorElement a3) (TensorElement a4))) x1 x2 x3 x4)

  zipWith5Op f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) =
    Tensor (A.zipWith5 (\a1 a2 a3 a4 a5 -> unTensorElement (f (TensorElement a1) (TensorElement a2) (TensorElement a3) (TensorElement a4) (TensorElement a5))) x1 x2 x3 x4 x5)

  zipWith6Op f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) =
    Tensor (A.zipWith6 (\a1 a2 a3 a4 a5 a6 -> unTensorElement (f (TensorElement a1) (TensorElement a2) (TensorElement a3) (TensorElement a4) (TensorElement a5) (TensorElement a6))) x1 x2 x3 x4 x5 x6)

  zipWith7Op f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7) =
    Tensor (A.zipWith7 (\a1 a2 a3 a4 a5 a6 a7 -> unTensorElement (f (TensorElement a1) (TensorElement a2) (TensorElement a3) (TensorElement a4) (TensorElement a5) (TensorElement a6) (TensorElement a7))) x1 x2 x3 x4 x5 x6 x7)

  zipWith8Op f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7) (Tensor x8) =
    Tensor (A.zipWith8 (\a1 a2 a3 a4 a5 a6 a7 a8 -> unTensorElement (f (TensorElement a1) (TensorElement a2) (TensorElement a3) (TensorElement a4) (TensorElement a5) (TensorElement a6) (TensorElement a7) (TensorElement a8))) x1 x2 x3 x4 x5 x6 x7 x8)

  zipWith9Op f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7) (Tensor x8) (Tensor x9) =
    Tensor (A.zipWith9 (\a1 a2 a3 a4 a5 a6 a7 a8 a9 -> unTensorElement (f (TensorElement a1) (TensorElement a2) (TensorElement a3) (TensorElement a4) (TensorElement a5) (TensorElement a6) (TensorElement a7) (TensorElement a8) (TensorElement a9))) x1 x2 x3 x4 x5 x6 x7 x8 x9)

  unzipOp (Tensor input) =
    case A.unzip input of
      (x1, x2) -> (Tensor x1, Tensor x2)

  unzip3Op (Tensor input) =
    case A.unzip3 input of
      (x1, x2, x3) -> (Tensor x1, Tensor x2, Tensor x3)

  unzip4Op (Tensor input) =
    case A.unzip4 input of
      (x1, x2, x3, x4) -> (Tensor x1, Tensor x2, Tensor x3, Tensor x4)

  unzip5Op (Tensor input) =
    case A.unzip5 input of
      (x1, x2, x3, x4, x5) -> (Tensor x1, Tensor x2, Tensor x3, Tensor x4, Tensor x5)

  unzip6Op (Tensor input) =
    case A.unzip6 input of
      (x1, x2, x3, x4, x5, x6) -> (Tensor x1, Tensor x2, Tensor x3, Tensor x4, Tensor x5, Tensor x6)

  unzip7Op (Tensor input) =
    case A.unzip7 input of
      (x1, x2, x3, x4, x5, x6, x7) -> (Tensor x1, Tensor x2, Tensor x3, Tensor x4, Tensor x5, Tensor x6, Tensor x7)

  unzip8Op (Tensor input) =
    case A.unzip8 input of
      (x1, x2, x3, x4, x5, x6, x7, x8) -> (Tensor x1, Tensor x2, Tensor x3, Tensor x4, Tensor x5, Tensor x6, Tensor x7, Tensor x8)

  unzip9Op (Tensor input) =
    case A.unzip9 input of
      (x1, x2, x3, x4, x5, x6, x7, x8, x9) -> (Tensor x1, Tensor x2, Tensor x3, Tensor x4, Tensor x5, Tensor x6, Tensor x7, Tensor x8, Tensor x9)

-- | Transform each element, preserving the shape while allowing a new element type.
map ::
  forall {t} sh a b.
  (Elementwise t, Shape sh, DType a, DType b) =>
  (Element t a -> Element t b) ->
  t sh a ->
  t sh b
map = mapOp @t @sh @a @b

-- | Pair corresponding elements of two tensors with the same shape.
zip ::
  forall {t} sh a b.
  (Elementwise t, Shape sh, DType a, DType b) =>
  t sh a ->
  t sh b ->
  t sh (a, b)
zip = zipOp @t @sh @a @b

-- | Group corresponding elements of three equally shaped tensors into triples.
-- The three element types may differ.
zip3 ::
  forall {t} sh a b c.
  (Elementwise t, Shape sh, DType a, DType b, DType c) =>
  t sh a ->
  t sh b ->
  t sh c ->
  t sh (a, b, c)
zip3 = zip3Op @t @sh @a @b @c

-- | Group corresponding elements of 4 equally shaped tensors into tuples.
-- Input element types may differ.
zip4 ::
  forall {t} sh a b c d.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d) =>
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh (a, b, c, d)
zip4 = zip4Op @t @sh @a @b @c @d

-- | Group corresponding elements of 5 equally shaped tensors into tuples.
-- Input element types may differ.
zip5 ::
  forall {t} sh a b c d e.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e) =>
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh (a, b, c, d, e)
zip5 = zip5Op @t @sh @a @b @c @d @e

-- | Group corresponding elements of 6 equally shaped tensors into tuples.
-- Input element types may differ.
zip6 ::
  forall {t} sh a b c d e f.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f) =>
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f ->
  t sh (a, b, c, d, e, f)
zip6 = zip6Op @t @sh @a @b @c @d @e @f

-- | Group corresponding elements of 7 equally shaped tensors into tuples.
-- Input element types may differ.
zip7 ::
  forall {t} sh a b c d e f g.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g) =>
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f ->
  t sh g ->
  t sh (a, b, c, d, e, f, g)
zip7 = zip7Op @t @sh @a @b @c @d @e @f @g

-- | Group corresponding elements of 8 equally shaped tensors into tuples.
-- Input element types may differ.
zip8 ::
  forall {t} sh a b c d e f g h.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h) =>
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f ->
  t sh g ->
  t sh h ->
  t sh (a, b, c, d, e, f, g, h)
zip8 = zip8Op @t @sh @a @b @c @d @e @f @g @h

-- | Group corresponding elements of 9 equally shaped tensors into tuples.
-- Input element types may differ.
zip9 ::
  forall {t} sh a b c d e f g h i.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i) =>
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f ->
  t sh g ->
  t sh h ->
  t sh i ->
  t sh (a, b, c, d, e, f, g, h, i)
zip9 = zip9Op @t @sh @a @b @c @d @e @f @g @h @i

-- | Combine corresponding elements of two tensors with the same shape.
zipWith ::
  forall {t} sh a b c.
  (Elementwise t, Shape sh, DType a, DType b, DType c) =>
  (Element t a -> Element t b -> Element t c) ->
  t sh a ->
  t sh b ->
  t sh c
zipWith = zipWithOp @t @sh @a @b @c

-- | Combine corresponding elements of three equally shaped tensors.
-- All input element types and the output element type may differ.
zipWith3 ::
  forall {t} sh a b c d.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d) =>
  (Element t a -> Element t b -> Element t c -> Element t d) ->
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d
zipWith3 = zipWith3Op @t @sh @a @b @c @d

-- | Combine corresponding elements of 4 equally shaped tensors.
-- All input element types and the output element type may differ.
zipWith4 ::
  forall {t} sh a b c d e.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e) =>
  (Element t a -> Element t b -> Element t c -> Element t d -> Element t e) ->
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e
zipWith4 = zipWith4Op @t @sh @a @b @c @d @e

-- | Combine corresponding elements of 5 equally shaped tensors.
-- All input element types and the output element type may differ.
zipWith5 ::
  forall {t} sh a b c d e f.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f) =>
  (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f) ->
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f
zipWith5 = zipWith5Op @t @sh @a @b @c @d @e @f

-- | Combine corresponding elements of 6 equally shaped tensors.
-- All input element types and the output element type may differ.
zipWith6 ::
  forall {t} sh a b c d e f g.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g) =>
  (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f -> Element t g) ->
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f ->
  t sh g
zipWith6 = zipWith6Op @t @sh @a @b @c @d @e @f @g

-- | Combine corresponding elements of 7 equally shaped tensors.
-- All input element types and the output element type may differ.
zipWith7 ::
  forall {t} sh a b c d e f g h.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h) =>
  (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f -> Element t g -> Element t h) ->
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f ->
  t sh g ->
  t sh h
zipWith7 = zipWith7Op @t @sh @a @b @c @d @e @f @g @h

-- | Combine corresponding elements of 8 equally shaped tensors.
-- All input element types and the output element type may differ.
zipWith8 ::
  forall {t} sh a b c d e f g h i.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i) =>
  (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f -> Element t g -> Element t h -> Element t i) ->
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f ->
  t sh g ->
  t sh h ->
  t sh i
zipWith8 = zipWith8Op @t @sh @a @b @c @d @e @f @g @h @i

-- | Combine corresponding elements of 9 equally shaped tensors.
-- All input element types and the output element type may differ.
zipWith9 ::
  forall {t} sh a b c d e f g h i j.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i, DType j) =>
  (Element t a -> Element t b -> Element t c -> Element t d -> Element t e -> Element t f -> Element t g -> Element t h -> Element t i -> Element t j) ->
  t sh a ->
  t sh b ->
  t sh c ->
  t sh d ->
  t sh e ->
  t sh f ->
  t sh g ->
  t sh h ->
  t sh i ->
  t sh j
zipWith9 = zipWith9Op @t @sh @a @b @c @d @e @f @g @h @i @j

-- | Split a tensor of 2-tuples into 2 tensor expressions with the same shape.
-- Component element types may differ and no computation is executed on the host.
unzip ::
  forall {t} sh a b.
  (Elementwise t, Shape sh, DType a, DType b) =>
  t sh (a, b) ->
  (t sh a, t sh b)
unzip = unzipOp @t @sh @a @b

-- | Split a tensor of 3-tuples into 3 tensor expressions with the same shape.
-- Component element types may differ and no computation is executed on the host.
unzip3 ::
  forall {t} sh a b c.
  (Elementwise t, Shape sh, DType a, DType b, DType c) =>
  t sh (a, b, c) ->
  (t sh a, t sh b, t sh c)
unzip3 = unzip3Op @t @sh @a @b @c

-- | Split a tensor of 4-tuples into 4 tensor expressions with the same shape.
-- Component element types may differ and no computation is executed on the host.
unzip4 ::
  forall {t} sh a b c d.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d) =>
  t sh (a, b, c, d) ->
  (t sh a, t sh b, t sh c, t sh d)
unzip4 = unzip4Op @t @sh @a @b @c @d

-- | Split a tensor of 5-tuples into 5 tensor expressions with the same shape.
-- Component element types may differ and no computation is executed on the host.
unzip5 ::
  forall {t} sh a b c d e.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e) =>
  t sh (a, b, c, d, e) ->
  (t sh a, t sh b, t sh c, t sh d, t sh e)
unzip5 = unzip5Op @t @sh @a @b @c @d @e

-- | Split a tensor of 6-tuples into 6 tensor expressions with the same shape.
-- Component element types may differ and no computation is executed on the host.
unzip6 ::
  forall {t} sh a b c d e f.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f) =>
  t sh (a, b, c, d, e, f) ->
  (t sh a, t sh b, t sh c, t sh d, t sh e, t sh f)
unzip6 = unzip6Op @t @sh @a @b @c @d @e @f

-- | Split a tensor of 7-tuples into 7 tensor expressions with the same shape.
-- Component element types may differ and no computation is executed on the host.
unzip7 ::
  forall {t} sh a b c d e f g.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g) =>
  t sh (a, b, c, d, e, f, g) ->
  (t sh a, t sh b, t sh c, t sh d, t sh e, t sh f, t sh g)
unzip7 = unzip7Op @t @sh @a @b @c @d @e @f @g

-- | Split a tensor of 8-tuples into 8 tensor expressions with the same shape.
-- Component element types may differ and no computation is executed on the host.
unzip8 ::
  forall {t} sh a b c d e f g h.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h) =>
  t sh (a, b, c, d, e, f, g, h) ->
  (t sh a, t sh b, t sh c, t sh d, t sh e, t sh f, t sh g, t sh h)
unzip8 = unzip8Op @t @sh @a @b @c @d @e @f @g @h

-- | Split a tensor of 9-tuples into 9 tensor expressions with the same shape.
-- Component element types may differ and no computation is executed on the host.
unzip9 ::
  forall {t} sh a b c d e f g h i.
  (Elementwise t, Shape sh, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i) =>
  t sh (a, b, c, d, e, f, g, h, i) ->
  (t sh a, t sh b, t sh c, t sh d, t sh e, t sh f, t sh g, t sh h, t sh i)
unzip9 = unzip9Op @t @sh @a @b @c @d @e @f @g @h @i

-- | Boolean operations on element expressions and equally shaped masks.
class Boolean a where
  -- | Negate each boolean value.
  not :: a -> a

  -- | Conjunction of corresponding boolean values.
  (&&) :: a -> a -> a

  -- | Disjunction of corresponding boolean values.
  (||) :: a -> a -> a

infixr 3 &&

infixr 2 ||

instance Boolean (TensorElement Bool) where
  not (TensorElement x) = TensorElement (A.not x)
  TensorElement x && TensorElement y = TensorElement (x A.&& y)
  TensorElement x || TensorElement y = TensorElement (x A.|| y)

instance (Shape sh) => Boolean (Tensor sh Bool) where
  not = map not
  (&&) = zipWith (&&)
  (||) = zipWith (||)

-- | Conditional selection of element expressions or tensor computations.
class Conditional a where
  -- | Boolean element or equally shaped tensor mask used by 'select'.
  type Condition a

  -- | Scalar boolean expression used to choose an entire computation.
  type Predicate a

  -- | Choose the first value where the condition is true and the second where it is false.
  -- Tensor masks select elementwise and all three tensor shapes must match.
  select :: Condition a -> a -> a -> a

  -- | Choose one branch using a scalar condition.
  -- For tensors this branches on the whole computation rather than selecting individual elements.
  cond :: Predicate a -> a -> a -> a

instance (DType a) => Conditional (TensorElement a) where
  type Condition (TensorElement a) = TensorElement Bool
  type Predicate (TensorElement a) = TensorElement Bool
  select = cond
  cond (TensorElement flag) (TensorElement x) (TensorElement y) = TensorElement (A.cond flag x y)

instance (Shape sh, DType a) => Conditional (Tensor sh a) where
  type Condition (Tensor sh a) = Tensor sh Bool
  type Predicate (Tensor sh a) = Tensor '[] Bool
  select = zipWith3 select
  cond (Tensor flag) (Tensor x) (Tensor y) = Tensor (A.acond (A.the flag) x y)

-- | Equality comparisons that produce element expressions or boolean tensors.
class Eq a where
  -- | Shape-preserving boolean result of a comparison.
  type Comparison a

  -- | Compare corresponding values for equality.
  (==) :: a -> a -> Comparison a

  -- | Compare corresponding values for inequality.
  (/=) :: a -> a -> Comparison a

infix 4 ==, /=

-- | Adapt a backend predicate to boolean element expressions.
elementComparison :: (A.Exp a -> A.Exp a -> A.Exp Bool) -> TensorElement a -> TensorElement a -> TensorElement Bool
elementComparison f (TensorElement x) (TensorElement y) = TensorElement (f x y)

instance (A.Eq a) => Eq (TensorElement a) where
  type Comparison (TensorElement a) = TensorElement Bool
  (==) = elementComparison (A.==)
  x /= y = not (x == y)

-- | Ordered comparisons and elementwise extrema without executing the computation.
-- Floating point comparisons follow IEEE semantics and extrema propagate NaN.
class (Eq a) => Ord a where
  -- | Compare corresponding values using strict less-than.
  (<) :: a -> a -> Comparison a

  -- | Compare corresponding values using less-than or equality.
  (<=) :: a -> a -> Comparison a

  -- | Compare corresponding values using strict greater-than.
  (>) :: a -> a -> Comparison a

  -- | Compare corresponding values using greater-than or equality.
  (>=) :: a -> a -> Comparison a

  -- | Select the smaller corresponding value.
  min :: a -> a -> a

  -- | Select the larger corresponding value.
  max :: a -> a -> a

infix 4 <, <=, >, >=

instance {-# OVERLAPPABLE #-} (A.Ord a) => Ord (TensorElement a) where
  (<) = elementComparison (A.<)
  (<=) = elementComparison (A.<=)
  (>) = elementComparison (A.>)
  (>=) = elementComparison (A.>=)
  min (TensorElement x) (TensorElement y) = TensorElement (A.min x y)
  max (TensorElement x) (TensorElement y) = TensorElement (A.max x y)

-- | Boolean ordering uses False before True.
instance {-# OVERLAPPING #-} Ord (TensorElement Bool) where
  x < y = not x && y
  x <= y = not x || y
  x > y = x && not y
  x >= y = x || not y
  min = (&&)
  max = (||)

-- | Preserve NaN before applying a floating point extremum operation.
nanExtremum :: (A.RealFloat a) => (A.Exp a -> A.Exp a -> A.Exp a) -> TensorElement a -> TensorElement a -> TensorElement a
nanExtremum f (TensorElement x) (TensorElement y) =
  TensorElement (A.cond (A.isNaN x A.|| A.isNaN y) (0 / 0) (f x y))

instance {-# OVERLAPPING #-} Ord (TensorElement Float) where
  (<) = elementComparison (A.<)
  (<=) = elementComparison (A.<=)
  (>) = elementComparison (A.>)
  (>=) = elementComparison (A.>=)
  min = nanExtremum A.min
  max = nanExtremum A.max

instance {-# OVERLAPPING #-} Ord (TensorElement Double) where
  (<) = elementComparison (A.<)
  (<=) = elementComparison (A.<=)
  (>) = elementComparison (A.>)
  (>=) = elementComparison (A.>=)
  min = nanExtremum A.min
  max = nanExtremum A.max

instance (Shape sh, DType a, Eq (TensorElement a)) => Eq (Tensor sh a) where
  type Comparison (Tensor sh a) = Tensor sh Bool
  (==) = zipWith (==)
  (/=) = zipWith (/=)

instance (Shape sh, DType a, Ord (TensorElement a)) => Ord (Tensor sh a) where
  (<) = zipWith (<)
  (<=) = zipWith (<=)
  (>) = zipWith (>)
  (>=) = zipWith (>=)
  min = zipWith min
  max = zipWith max

-- | Integer division with truncation toward zero and its corresponding remainder.
-- Division by zero is undefined and must be avoided by the caller.
class Integral a where
  -- | Divide corresponding values, rounding the quotient toward zero.
  quot :: a -> a -> a

  -- | Remainder whose sign follows the dividend.
  rem :: a -> a -> a

instance (A.Integral a) => Integral (TensorElement a) where
  quot (TensorElement x) (TensorElement y) = TensorElement (A.quot x y)
  rem (TensorElement x) (TensorElement y) = TensorElement (A.rem x y)

instance (Shape sh, DType a, Integral (TensorElement a)) => Integral (Tensor sh a) where
  quot = zipWith quot
  rem = zipWith rem

-- | Classification of floating point elements and tensors.
class (Eq a) => RealFloat a where
  -- | Test whether each value is NaN.
  isNaN :: a -> Comparison a

  -- | Test whether each value is positive or negative infinity.
  isInfinite :: a -> Comparison a

instance (A.RealFloat a) => RealFloat (TensorElement a) where
  isNaN (TensorElement x) = TensorElement (A.isNaN x)
  isInfinite (TensorElement x) = TensorElement (A.isInfinite x)

instance (Shape sh, DType a, RealFloat (TensorElement a)) => RealFloat (Tensor sh a) where
  isNaN = map isNaN
  isInfinite = map isInfinite
