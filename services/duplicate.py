import imagehash
from PIL import Image
import io
from typing import List, Tuple

DUPLICATE_THRESHOLD = 8  # Hamming distance threshold

def get_hash(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    ahash = str(imagehash.average_hash(img, hash_size=16))
    dhash = str(imagehash.dhash(img, hash_size=16))
    return f"{ahash}:{dhash}"

def hamming_distance(hash1: str, hash2: str) -> int:
    a1, d1 = hash1.split(':')
    a2, d2 = hash2.split(':')
    dist_a = imagehash.hex_to_hash(a1) - imagehash.hex_to_hash(a2)
    dist_d = imagehash.hex_to_hash(d1) - imagehash.hex_to_hash(d2)
    return (dist_a + dist_d) // 2

def check_duplicates(
    new_hashes: List[str],
    existing_listings: List[dict]
) -> dict:
    for new_hash in new_hashes:
        for listing in existing_listings:
            for stored_hash in (listing.get('image_hashes') or []):
                dist = hamming_distance(new_hash, stored_hash)
                if dist <= DUPLICATE_THRESHOLD:
                    confidence = round(1 - (dist / DUPLICATE_THRESHOLD), 2)
                    return {
                        'is_duplicate': True,
                        'duplicate_listing_id': str(listing['id']),
                        'confidence': max(0.0, confidence),
                        'hashes': new_hashes,
                    }
    return {
        'is_duplicate': False,
        'duplicate_listing_id': None,
        'confidence': 1.0,
        'hashes': new_hashes,
    }
