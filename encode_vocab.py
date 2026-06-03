from object_detection.encoder import CLIPEncoder
from object_detection.vocab import encode_and_cache

encoder = CLIPEncoder()
encode_and_cache(encoder)
