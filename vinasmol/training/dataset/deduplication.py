from datatrove.pipeline.base import PipelineStep, DocumentsPipeline
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document

from rensa import RMinHash, RMinHashLSH

# TODO: Possibly apply stronger rules for deduplication, such as:
# - Linewise filtering: Read more, sign-in, items in cart... (see RefinedWeb G.2)

# https://github.com/beowolx/rensa?tab=readme-ov-file#deduplicating-with-rminhashlsh

class RensaBuildIndex(PipelineStep):
    """Build a MinHashLSH index with Rensa as a backend."""

    name = "🗂️ Rensa deduplication stage 1"

    def __init__(
            self,
            num_perm=128,
            seed=20250801,
            lsh_threshold: float = 0.8,
            num_bands: int = 16,
            final_jaccard_threshold: float = 0.85
        ):
        if num_perm % num_bands != 0:
            raise ValueError(f"num_bands ({num_bands}) must divide num_perm ({num_perm}).")
    
        super().__init__()
        self.num_perm = num_perm
        self.seed = seed
        self.lsh_threshold = lsh_threshold
        self.num_bands = num_bands
        self.final_jaccard_threshold = final_jaccard_threshold
        self._minhashes = {}
        self._lsh_index = RMinHashLSH(
            threshold=lsh_threshold,
            num_perm=num_perm,
            num_bands=num_bands,
        )
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if data:
            for document in data:
                self.index(document)
                yield document
    
    def index(self, document: Document):
        rminhash_obj = self.generate_minhash_signature(document.text)
        self._minhashes[document.id] = rminhash_obj
        self._lsh_index.insert(document.id, rminhash_obj)
    
    # Define a function to generate MinHash (works for RMinHash, CMinHash)
    def generate_minhash_signature(self, text: str):
        m = RMinHash(num_perm=self.num_perm, seed=self.seed)
        m.update(text.split())
        return m


class RensaDeduplicate(BaseFilter):
    """Deduplicate documents with a MinHashLSH index implemented by Rensa.

    Deduplication is single-threaded and not distributed, which is sufficient for our scale.
    See real-world benchmarks [here](https://github.com/beowolx/rensa?tab=readme-ov-file#large-scale-benchmark-salesforcewikitext-18-million-rows).
    """

    name = "🦀 Rensa deduplication stage 2"

    def __init__(self, rensa_index: RensaBuildIndex, exclusion_writer = None):
        """Initialize the pipeline step.

        Args:
            rensa_index (RensaBuildIndex): Must be the previous pipeline step.
            exclusion_writer (DiskWriter, optional): Exclusion writer.
        """
        super().__init__(exclusion_writer, batch_size=1)
        self.rensa_index = rensa_index
        self._to_remove = set()
        self._num_visited = 0
    
    def clear(self):
        del self._to_remove
        del self.rensa_index._lsh_index
        del self.rensa_index._minhashes
    
    def increment_counter(self):
        """Free memory if the pipeline step is finished."""
        self._num_visited += 1
        if self._num_visited == len(self.rensa_index._minhashes):
            self.clear()

    def filter(self, document: Document) -> bool | tuple[bool, str]:
        if document.id in self._to_remove:
            self.increment_counter()
            return False, "duplicate"

        query_minhash = self.rensa_index._minhashes[document.id]
        candidate_ids = self.rensa_index._lsh_index.query(query_minhash)

        for candidate_id in candidate_ids:
            if candidate_id == document.id or candidate_id in self._to_remove:
                continue
            
            candidate_minhash = self.rensa_index._minhashes[candidate_id]
            actual_jaccard = query_minhash.jaccard(candidate_minhash)

            if actual_jaccard >= self.rensa_index.final_jaccard_threshold:
                # Keep the item with the smaller original index
                if document.id < candidate_id:
                    self._to_remove.add(candidate_id)
                else:
                    self._to_remove.add(document.id)
                    self.increment_counter()
                    return False, "duplicate"

        self.increment_counter()
        return True