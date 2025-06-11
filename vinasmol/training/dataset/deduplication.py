from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document

from rensa import RMinHash

# TODO: Possibly apply stronger rules for deduplication, such as:
# - Linewise filtering: Read more, sign-in, items in cart... (see RefinedWeb G.2)
# - Possibly rely on paragraph deduplication for privacy policy / cookies copypasta


# Wikipedia-specific
class RensaDeduplication(BaseFilter):
    """Deduplicate documents with MinHash with Rensa as a backend.

    Deduplication is single-threaded and not distributed, which is sufficient for our scale.
    See real-world benchmarks [here](https://github.com/beowolx/rensa?tab=readme-ov-file#large-scale-benchmark-salesforcewikitext-18-million-rows).
    """
    def __init__(self, minhash_class=RMinHash, num_perm=128, seed=20250801):
        super().__init__()
        self._minhash_class = minhash_class
        self._num_perm = num_perm
        self._seed = seed
        self._unique_hashes = set()

    def filter(self, document: Document) -> bool | tuple[bool, str]:
        minhash_obj = self.generate_minhash_signature(document.text)
        hash_tuple = tuple(minhash_obj.digest())
    
        if hash_tuple not in self._unique_hashes:
            self._unique_hashes.add(hash_tuple)
            return True
        else:
            return False, "duplicate"  

    # Define a function to generate MinHash (works for RMinHash, CMinHash)
    def generate_minhash_signature(self, text: str):
        m = self._minhash_class(num_perm=self._num_perm, seed=self._seed)
        m.update(text.split())
        return m
