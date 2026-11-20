{-# LANGUAGE TypeFamilies #-}

-- | Concatenation and splitting of heterogeneous input records.
module Synapse.Autograd.Record
    ( type (++)
    , append
    , split
    ) where

import Numeric.Backprop.Op (Rec (..))

-- | Concatenate the types of two records.
type family xs ++ ys where
    '[] ++ ys = ys
    (x ': xs) ++ ys = x ': (xs ++ ys)

-- | Append records in field order.
append
    :: Rec f xs
    -> Rec f ys
    -> Rec f (xs ++ ys)
append RNil ys = ys
append (x :& xs) ys = x :& append xs ys

-- | Split a record using another record to identify the left fields.
split
    :: Rec g xs
    -> Rec f (xs ++ ys)
    -> (Rec f xs, Rec f ys)
split RNil values = (RNil, values)
split (_ :& xs) (x :& values) = let (left, right) = split xs values in (x :& left, right)
