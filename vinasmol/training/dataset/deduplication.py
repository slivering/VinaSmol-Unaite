import glob
from pathlib import Path
import subprocess
from subprocess import CalledProcessError
import tempfile

from datatrove.pipeline.base import PipelineStep, DocumentsPipeline
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.data import Document
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import ExtensionHelperES as EH

from rensa import RMinHash, RMinHashLSH

# TODO: Possibly apply stronger rules for deduplication, such as:
# - Linewise filtering: Read more, sign-in, items in cart... (see RefinedWeb G.2)

# https://github.com/beowolx/rensa?tab=readme-ov-file#deduplicating-with-rminhashlsh

class RensaBuildIndex(PipelineStep):
    """Build a MinHashLSH index with Rensa as a backend.
    
    For now, hash computation is single-threaded, without merging results.
    """

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
        if rank != 0:
            raise RuntimeError("Cannot run RensaBuildIndex with multiple workers")
        if data:
            for document in data:
                assert hasattr(self, "_minhashes")
                self.index(document)
                if not hasattr(self, "_minhashes"):
                    raise RuntimeError("WTF0")
                yield document
                if not hasattr(self, "_minhashes"):
                    raise RuntimeError("WTF1")
    
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
        logger.warning("Clearing Rensa index to free memory")
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


class ESComputeRangesExternal(PipelineStep):
    """STAGE 2.5 of exact substring deduplication

    Runs the external scripts from https://github.com/google-research/deduplicate-text-datasets
    submodule. Requires Rust to be installed.
    """

    type = "🫂 - DEDUP"
    name = "🦀 - exact-substrings stage 2.5 (compute ranges in Rust)"

    def __init__(
            self,
            length_threshold: int = 100,
            data_folder: DataFolderLike = None,
            tmp_dir: Path = None,
            google_repo_path: Path = None,
            #cargo_path: Path = None,
            num_threads: int = 8,
        ):
        super().__init__()
        self.data_folder = get_datafolder(data_folder)
        if google_repo_path is None:
            google_repo_path = Path(__file__).parent / "deduplicate-text-datasets"
        self.google_repo_path = google_repo_path
        self._tmp_dir = tmp_dir

        self.length_threshold = length_threshold
        self.num_threads = num_threads

    def setup(self):
        self.tmp_dir = tempfile.TemporaryDirectory(
            prefix="deduplicate-text-datasets",
            dir=self._tmp_dir,
        )
        try:
            with self.track_time(unit="cargo_build"):
                cargo_build = subprocess.run(
                    # TODO: Cargo.lock with old edition
                    # TODO: patch the script to be able to run with --release
                    ["cargo", "build"],
                    capture_output=True,
                    check=True,
                    cwd=self.google_repo_path,
                )
            logger.info(f"Successfully built {self.google_repo_path}")
            logger.info(f"Got cargo output:\n{cargo_build.stdout.decode()}")
        except CalledProcessError as e:
            logger.critical(
                "cargo command might not be installed or accessible, "
                "or cargo build failed.\n{}",
                e.stderr.decode()
            )
            raise e

        subprocess.run(
            ["ln", "-s", "-f", self.tmp_dir.name, "./tmp"],
            capture_output=True,
            check=True,
            cwd=self.google_repo_path,
        )

    def run(self, data, rank = 0, world_size = 1):
        if rank != 0:
            raise ValueError(
                "deduplicate-text-datasets is already parallelized and doesn't need "
                "to be run with multiple Python workers"
            )
        self.setup()

        dataset_file = f"dataset{EH.stage_2_big_sequence}"
        byterange_file = dataset_file.replace(EH.stage_2_big_sequence, EH.stage_3_bytes_ranges)

        if not self.data_folder.exists(dataset_file):
            raise RuntimeError(
                f"Missing {self.data_folder.path}/{dataset_file}. "
                "Did you run the `ESMergeSequences` block in the pipeline before?"
            )
        dataset_file = str(Path(self.data_folder.path) / dataset_file)

        with self.track_time(unit="make_suffix_array"):
            try:
                make_suffix_array = subprocess.run(
                    ["python", "scripts/make_suffix_array.py", dataset_file],
                    capture_output=True,
                    check=True,
                    cwd=self.google_repo_path,
                )
                logger.info(
                    "Got output for make_suffix_array:\n"
                    f"{make_suffix_array.stdout.decode()}"
                )
            except CalledProcessError as e:
                logger.critical("make_suffix_array.py failed\n{}", e.stderr.decode())
                raise e

        # Cleaning up manually because the previous script fails
        temp_big_sequences = glob.glob("out.table.bin.*", root_dir=self.tmp_dir.name)
        #for file in temp_big_sequences:
        #    Path(file).unlink(missing_ok=True)
        
        try:
            with self.track_time(unit="self_similar"):
                self_similar = subprocess.run(
                    [
                        "cargo", "run", "self-similar",
                        "--data-file", dataset_file,
                        "--length-threshold", str(self.length_threshold),
                        "--cache-dir", self.tmp_dir.name,
                        "--num-threads", str(self.num_threads),
                    ],
                    capture_output=True,
                    check=True,
                    cwd=self.google_repo_path,
                )
                logger.info(
                    "Got output for self-similar:\n"
                    f"{self_similar.stdout.decode()}"
                )
            with self.track_time(unit="collect"):
                with self.data_folder.open(byterange_file, "wb") as f:
                    subprocess.run(
                        [
                            "cargo", "run", "collect",
                            "--data-file", dataset_file,
                            "--length-threshold", str(self.length_threshold),
                            "--cache-dir", self.tmp_dir.name,
                        ],
                        check=True,
                        cwd=self.google_repo_path,
                        stdout=f
                    )
                    logger.info("Done collecting self-similarity")
        except CalledProcessError as e:
            logger.critical("self-similar failed\n{}", e.stderr.decode())
            raise e
        
        temp_es_sequences = glob.glob(
            f"*.{EH.stage_1_sequence}*",
            root_dir=self.data_folder.path,
        )
        temp_big_sequences = glob.glob(
            f"dataset{EH.stage_2_big_sequence}.*",
            root_dir=self.data_folder.path,
        )
        self.data_folder.move(f"00000{EH.stage_1_sequence}", f"dataset{EH.stage_1_sequence}")
        self.data_folder.move(f"00000{EH.stage_1_sequence_size}", f"dataset{EH.stage_1_sequence_size}")

        #for file in temp_es_sequences + temp_big_sequences:
        #    Path(file).unlink(missing_ok=True)

        #self.tmp_dir.cleanup()
        (Path(self.google_repo_path) / "tmp").unlink()
        logger.info("Done")
