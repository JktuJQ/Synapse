{- | Read and write <https://github.com/safetensors/safetensors safetensors> weights.

@
import qualified Synapse.NN.Weights as Weights
import qualified Synapse.NN.Weights.Safetensors as Safetensors

Safetensors.writeSafetensors "weights.safetensors" (Weights.extractWeights parameters)
@

Ordinary field paths use dotted names such as @encoder.denseWeights@.
Ambiguous paths receive unique alternative names and a string metadata entry
preserving their original components. The empty name represents the root tensor.
External dotted names are split into path components; other characters are literal.
Use 'Map.mapKeys' and 'Map.traverseWithKey' to adapt external conventions before loading.

Supported element types are 'Float', 'Double', 'Int', 'Int32', 'Int64',
'Word32', 'Word64', and 'Bool'. Host 'Int' is stored as I32 or I64 with
a string metadata entry recording its paths, preserving its Haskell type
when read by Synapse. Other file metadata is ignored. Unsupported dtypes
are reported without implicit conversion.
-}
module Synapse.NN.Weights.Safetensors
    ( SafetensorsError (..)
    , encodeSafetensors
    , decodeSafetensors
    , readSafetensors
    , writeSafetensors
    )
where

import Control.Monad (foldM, foldM_, unless, when)
import Data.Binary.Get (getLazyByteString, getWord64le, runGetOrFail)
import Data.Bits (finiteBitSize, shiftL, (.|.))
import qualified Data.ByteString as BS
import qualified Data.ByteString.Builder as Builder
import qualified Data.ByteString.Lazy as LBS
import Data.Char (chr, ord)
import Data.Foldable (traverse_)
import Data.Int (Int32, Int64)
import Data.List (intercalate, sortOn)
import qualified Data.Map.Strict as Map
import Data.Ratio (denominator, numerator)
import qualified Data.Set as Set
import qualified Data.Text as Text
import qualified Data.Text.Encoding as Text
import Data.Type.Equality ((:~:) (Refl))
import Data.Typeable (TypeRep, Typeable, eqT)
import Data.Word (Word32, Word64)
import GHC.Float (castWord32ToFloat, castWord64ToDouble)
import Synapse.NN.Weights
import Synapse.Tensor.Representation (DType)
import qualified Text.JSON as JSON

{- | File structure, dtype, Unicode, or weight construction errors.
File-system exceptions from 'readSafetensors' and 'writeSafetensors' remain ordinary IO exceptions.
-}
data SafetensorsError
    = -- | Invalid file header or metadata.
      InvalidHeader String
    | -- | Tensor name and invalid data description.
      InvalidTensor String String
    | -- | File dtype that cannot be decoded.
      UnsupportedDType String
    | -- | Element type that cannot be encoded.
      UnsupportedElementType TypeRep
    | -- | Invalid tensor data or duplicate field paths.
      InvalidWeights WeightsError
    deriving (Eq, Show)

-- | Element types supported by the file codec.
data FileDType = F32 | F64 | I32 | I64 | U32 | U64 | BOOL
    deriving (Bounded, Enum, Eq, Show)

-- | Parse a supported dtype name.
parseDType
    :: String
    -> Maybe FileDType
parseDType name = lookup name [(show dtype, dtype) | dtype <- [minBound .. maxBound]]

-- | Size of one encoded element in bytes.
dtypeWidth
    :: FileDType
    -> Int
dtypeWidth dtype = case dtype of
    F32 -> 4
    F64 -> 8
    I32 -> 4
    I64 -> 8
    U32 -> 4
    U64 -> 8
    BOOL -> 1

-- | A file dtype and its host element encoding.
data Encoding a = Encoding
    { encodingDType :: FileDType
    , encodeElement :: a -> Builder.Builder
    }

-- | Select the safetensors encoding for a supported element type.
encoding
    :: forall a
     . Typeable a
    => Maybe (Encoding a)
encoding
    | Just Refl <- eqT @a @Float = Just (Encoding F32 Builder.floatLE)
    | Just Refl <- eqT @a @Double = Just (Encoding F64 Builder.doubleLE)
    | Just Refl <- eqT @a @Int =
        Just $
            if finiteBitSize (0 :: Int) == 64
                then Encoding I64 (Builder.int64LE . fromIntegral)
                else Encoding I32 (Builder.int32LE . fromIntegral)
    | Just Refl <- eqT @a @Int32 = Just (Encoding I32 Builder.int32LE)
    | Just Refl <- eqT @a @Int64 = Just (Encoding I64 Builder.int64LE)
    | Just Refl <- eqT @a @Word32 = Just (Encoding U32 Builder.word32LE)
    | Just Refl <- eqT @a @Word64 = Just (Encoding U64 Builder.word64LE)
    | Just Refl <- eqT @a @Bool = Just (Encoding BOOL (Builder.word8 . fromIntegral . fromEnum))
    | otherwise = Nothing

{- | Encode names and JSON strings as Unicode scalar values.
Text.JSON decodes each escaped UTF-16 code unit separately.
-}
normalizeString
    :: String
    -> Either SafetensorsError String
normalizeString [] = Right []
normalizeString (high : low : rest)
    | h >= 0xD800 && h <= 0xDBFF && l >= 0xDC00 && l <= 0xDFFF =
        (chr (0x10000 + (h - 0xD800) * 0x400 + l - 0xDC00) :) <$> normalizeString rest
  where
    h = ord high
    l = ord low
normalizeString (value : rest)
    | ord value >= 0xD800 && ord value <= 0xDFFF = Left (InvalidHeader "unpaired Unicode surrogate")
    | otherwise = (value :) <$> normalizeString rest

-- | Normalize JSON strings and reject duplicate object keys.
normalizeJSON
    :: JSON.JSValue
    -> Either SafetensorsError JSON.JSValue
normalizeJSON (JSON.JSString value) = JSON.showJSON <$> normalizeString (JSON.fromJSString value)
normalizeJSON (JSON.JSArray values) = JSON.JSArray <$> traverse normalizeJSON values
normalizeJSON (JSON.JSObject fields) = do
    entries <-
        traverse (\(key, value) -> (,) <$> normalizeString key <*> normalizeJSON value) (JSON.fromJSObject fields)
    foldM_
        ( \seen (key, _) ->
            if Set.member key seen then Left (InvalidHeader ("duplicate key: " ++ key)) else Right (Set.insert key seen)
        )
        Set.empty
        entries
    pure $ JSON.makeObj entries
normalizeJSON value = Right value

-- | Decode and normalize a UTF-8 JSON value.
parseJSON
    :: BS.ByteString
    -> Either SafetensorsError JSON.JSValue
parseJSON bytes = do
    text <- either (Left . InvalidHeader . show) Right (Text.decodeUtf8' bytes)
    value <- either (Left . InvalidHeader) Right (JSON.resultToEither (JSON.decode (Text.unpack text)))
    normalizeJSON value

-- | Prefer conventional dotted names, keeping colliding or reserved paths unique.
namedWeights
    :: Weights
    -> [(String, FieldPath, WeightTensor)]
namedWeights = snd . foldr assign (Set.singleton "__metadata__", []) . Map.toAscList . weightsEntries
  where
    assign (path, tensor) (used, entries) =
        let preferred = intercalate "." path
            pointer = concatMap (("/" ++) . concatMap escape) path
            choose index =
                let candidate = pointer ++ "#" ++ show index
                 in if Set.member candidate used then choose (index + 1 :: Int) else candidate
            name
                | not (Set.member preferred used) = preferred
                | not (Set.member pointer used) = pointer
                | otherwise = choose 0
         in (Set.insert name used, (name, path, tensor) : entries)
    escape '~' = "~0"
    escape '/' = "~1"
    escape character = [character]

-- | Split a conventional tensor name into field path components.
fieldPath
    :: String
    -> FieldPath
fieldPath "" = []
fieldPath name = split '.' name

-- | Split a string on a separator, preserving empty components.
split
    :: Char
    -> String
    -> [String]
split separator input = case break (== separator) input of
    (part, []) -> [part]
    (part, _ : rest) -> part : split separator rest

-- | Encode Safetensors format.
encodeSafetensors
    :: Weights
    -> Either SafetensorsError LBS.ByteString
encodeSafetensors weights = do
    let named = namedWeights weights
        paths = [(name, JSON.showJSON path) | (name, path, _) <- named, fieldPath name /= path]
    (entries, _, body, hostInts) <- foldM encodeEntry ([], 0, mempty, []) named
    let annotations =
            [("synapse.host_int", JSON.showJSON (JSON.encode (reverse hostInts))) | not (null hostInts)]
                ++ [("synapse.field_paths", JSON.showJSON (JSON.encode (JSON.makeObj paths))) | not (null paths)]
        metadata = [("__metadata__", JSON.makeObj annotations) | not (null annotations)]
        headerValue = JSON.makeObj (reverse entries ++ metadata)
    normalized <- normalizeJSON headerValue
    let header = Text.encodeUtf8 $ Text.pack $ JSON.encode normalized
        padding = (8 - BS.length header `mod` 8) `mod` 8
        size = BS.length header + padding
    when (size > 100000000) $ Left (InvalidHeader "header exceeds 100 MB")
    pure $
        Builder.toLazyByteString $
            Builder.word64LE (fromIntegral size)
                <> Builder.byteString header
                <> Builder.byteString (BS.replicate padding 32)
                <> body
  where
    encodeEntry (entries, offset, body, hostInts) (name, _, tensor) = withWeightTensor tensor $ \(values :: WeightTensorView a) -> case encoding @a of
        Nothing -> Left (UnsupportedElementType (weightTensorType tensor))
        Just format -> do
            let count = product (weightTensorViewShape values)
                end = offset + toInteger count * toInteger (dtypeWidth $ encodingDType format)
                describe =
                    JSON.makeObj
                        [ ("dtype", JSON.showJSON (show $ encodingDType format))
                        , ("shape", JSON.showJSON (weightTensorViewShape values))
                        , ("data_offsets", JSON.showJSON [offset, end])
                        ]
                write index
                    | index == count = mempty
                    | otherwise = encodeElement format (weightTensorViewAt values index) <> write (index + 1)
                ints = case eqT @a @Int of
                    Just _ -> name : hostInts
                    Nothing -> hostInts
            when (end > toInteger (maxBound :: Word64)) $ Left (InvalidTensor name "data offset exceeds Word64")
            pure ((name, describe) : entries, end, body <> write 0, ints)

-- | Parsed tensor dimensions and a range relative to the data section.
data Entry = Entry
    { entryName :: String
    , entryDType :: String
    , entryDimensions :: [Int]
    , entryBegin :: Integer
    , entryEnd :: Integer
    }

-- | Extract the members of a JSON object.
object
    :: JSON.JSValue
    -> Either SafetensorsError [(String, JSON.JSValue)]
object (JSON.JSObject fields) = Right (JSON.fromJSObject fields)
object _ = Left (InvalidHeader "expected JSON object")

-- | Extract a JSON string.
string
    :: JSON.JSValue
    -> Either SafetensorsError String
string (JSON.JSString value) = Right (JSON.fromJSString value)
string _ = Left (InvalidHeader "expected JSON string")

-- | Extract a nonnegative JSON integer.
integer
    :: JSON.JSValue
    -> Either SafetensorsError Integer
integer (JSON.JSRational _ value)
    | denominator value == 1 && value >= 0 = Right (numerator value)
integer _ = Left (InvalidHeader "expected nonnegative integer")

-- | Parse each element of a JSON array.
array
    :: (JSON.JSValue -> Either SafetensorsError a)
    -> JSON.JSValue
    -> Either SafetensorsError [a]
array parse (JSON.JSArray values) = traverse parse values
array _ _ = Left (InvalidHeader "expected JSON array")

-- | Decode a tensor after checking its dtype, shape, and byte range.
decodeEntry
    :: Map.Map String FieldPath
    -> Set.Set String
    -> LBS.ByteString
    -> Entry
    -> Either SafetensorsError (FieldPath, WeightTensor)
decodeEntry paths hostInts buffer entry = do
    let name = entryName entry
        dtypeName = entryDType entry
        dimensions = entryDimensions entry
        begin = entryBegin entry
        end = entryEnd entry
        path = Map.findWithDefault (fieldPath name) name paths
    dtype <- maybe (Left $ UnsupportedDType dtypeName) Right (parseDType dtypeName)
    let count = product (map toInteger dimensions)
    when (count > toInteger (maxBound :: Int)) $ Left (InvalidTensor name "element count exceeds Int")
    unless (end - begin == count * toInteger (dtypeWidth dtype)) $
        Left (InvalidTensor name "byte count does not match shape and dtype")
    let bytes = LBS.toStrict $ LBS.take (fromInteger (end - begin)) $ LBS.drop (fromInteger begin) buffer
        word size index =
            foldr
                (\byte value -> value `shiftL` 8 .|. fromIntegral (BS.index bytes (index * size + byte)))
                0
                [0 .. size - 1]
                :: Word64
        build
            :: (DType a, Typeable a)
            => (Int -> a)
            -> Either SafetensorsError WeightTensor
        build readElement = either (Left . InvalidWeights) Right $ weightTensorFromView $ WeightTensorView dimensions readElement
        native = Set.member name hostInts
    when (native && dtype `notElem` [I32, I64]) $
        Left (InvalidTensor name "host Int metadata requires I32 or I64")
    tensor <- case dtype of
        F32 -> build (castWord32ToFloat . fromIntegral . word 4)
        F64 -> build (castWord64ToDouble . word 8)
        I32 | native -> build (fromIntegral . (fromIntegral :: Word64 -> Int32) . word 4 :: Int -> Int)
        I64 | native -> do
            let signed = fromIntegral . word 8 :: Int -> Int64
            when
                ( finiteBitSize (0 :: Int) < 64
                    && any
                        ( \index ->
                            toInteger (signed index) < toInteger (minBound :: Int)
                                || toInteger (signed index) > toInteger (maxBound :: Int)
                        )
                        [0 .. fromInteger count - 1]
                )
                $ Left (InvalidTensor name "host Int value is out of range")
            build (fromIntegral . signed :: Int -> Int)
        I32 -> build (fromIntegral . word 4 :: Int -> Int32)
        I64 -> build (fromIntegral . word 8 :: Int -> Int64)
        U32 -> build (fromIntegral . word 4 :: Int -> Word32)
        U64 -> build (word 8)
        BOOL -> do
            when (BS.any (> 1) bytes) $ Left (InvalidTensor name "BOOL data must contain zero or one")
            build ((/= 0) . BS.index bytes)
    pure (path, tensor)

-- | Decode Safetensors format.
decodeSafetensors
    :: LBS.ByteString
    -> Either SafetensorsError Weights
decodeSafetensors input = do
    (buffer, headerSize) <- case runGetOrFail getWord64le input of
        Left (_, _, message) -> Left (InvalidHeader message)
        Right (rest, _, size) -> Right (rest, size)
    when (headerSize == 0 || headerSize > 100000000) $
        Left (InvalidHeader "header size must be between 1 and 100 MB")
    (payload, header) <- case runGetOrFail (getLazyByteString (fromIntegral headerSize)) buffer of
        Left (_, _, message) -> Left (InvalidHeader message)
        Right (rest, _, bytes) -> Right (rest, LBS.toStrict bytes)
    unless (BS.head header == 123) $ Left (InvalidHeader "header must begin with an object")
    fields <- object =<< parseJSON header
    metadata <- maybe (Right []) object (lookup "__metadata__" fields)
    traverse_ (string . snd) metadata
    let annotation key fallback = case lookup key metadata of
            Nothing -> Right fallback
            Just encoded -> do
                content <- string encoded
                parseJSON (Text.encodeUtf8 (Text.pack content))
    hostInts <- Set.fromList <$> (array string =<< annotation "synapse.host_int" (JSON.JSArray []))
    rawPaths <- object =<< annotation "synapse.field_paths" (JSON.makeObj [])
    paths <- Map.fromList <$> traverse (\(name, value) -> (,) name <$> array string value) rawPaths
    let required fields' key = maybe (Left (InvalidHeader ("missing field: " ++ key))) Right (lookup key fields')
        parseEntry (name, value) = do
            description <- object value
            dtype <- string =<< required description "dtype"
            rawShape <- array integer =<< required description "shape"
            when (any (> toInteger (maxBound :: Int)) rawShape) $ Left (InvalidTensor name "dimension exceeds Int")
            offsets <- array integer =<< required description "data_offsets"
            case offsets of
                [begin, end] | begin <= end -> Right (Entry name dtype (map fromInteger rawShape) begin end)
                _ -> Left (InvalidTensor name "expected ordered [begin, end] offsets")
    entries <- traverse parseEntry (filter ((/= "__metadata__") . fst) fields)
    let names = Set.fromList $ map entryName entries
    unless (hostInts `Set.isSubsetOf` names) $ Left (InvalidHeader "host Int metadata names a missing tensor")
    unless (Map.keysSet paths `Set.isSubsetOf` names) $
        Left (InvalidHeader "field path metadata names a missing tensor")
    let available = toInteger (LBS.length payload)
        checkRange offset entry
            | begin /= offset = Left (InvalidTensor name "data ranges overlap or contain a gap")
            | end > available = Left (InvalidTensor name "data range exceeds file size")
            | otherwise = Right end
          where
            name = entryName entry
            begin = entryBegin entry
            end = entryEnd entry
    end <- foldM checkRange 0 (sortOn (\entry -> (entryBegin entry, entryEnd entry)) entries)
    unless (end == available) $ Left (InvalidHeader "unindexed trailing tensor data")
    decoded <- traverse (decodeEntry paths hostInts payload) entries
    let checkPath seen (path, _)
            | Set.member path seen = Left (InvalidWeights (DuplicateField path))
            | otherwise = Right (Set.insert path seen)
    foldM_ checkPath Set.empty decoded
    pure $ Weights $ Map.fromList decoded

-- | Read weights in Safetensors format.
readSafetensors
    :: FilePath
    -> IO (Either SafetensorsError Weights)
readSafetensors path = decodeSafetensors . LBS.fromStrict <$> BS.readFile path

-- | Write weights in Safetensors format.
writeSafetensors
    :: FilePath
    -> Weights
    -> IO (Either SafetensorsError ())
writeSafetensors path weights = case encodeSafetensors weights of
    Left problem -> pure (Left problem)
    Right bytes -> Right <$> LBS.writeFile path bytes
