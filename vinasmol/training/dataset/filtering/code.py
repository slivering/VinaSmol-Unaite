from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.stats import (
    DocStats, LineStats, TokenStats, 
    TopKConfig
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

from ..deduplication import RensaBuildIndex, RensaDeduplicate
from .common import JsonlShard

from . import (
    DATA_DIR_CODE, MAIN_OUTPUT_DIR, FILTERING_OUTPUT_DIR, FILTERING_REMOVED_DIR,
    LOGGING_DIR, STATS_DIR, SEED,
)
CORPUS = "code-all"

output_intermediate_1 = f"{FILTERING_OUTPUT_DIR}/output_1/{CORPUS}"
output_intermediate_2 = f"{FILTERING_OUTPUT_DIR}/output_2/{CORPUS}"
output_intermediate_3 = f"{FILTERING_OUTPUT_DIR}/output_3/{CORPUS}"

top_k_config = TopKConfig(top_k_groups=[], top_k=1)

main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            str(DATA_DIR_CODE),
            recursive=True,
            default_metadata={"dump": CORPUS},
        ),
        JsonlWriter(output_intermediate_1),
    ],
    tasks=4,
    workers=4,
    logging_dir=f"{LOGGING_DIR}/base_processing/{CORPUS}",
)


rensa_index = RensaBuildIndex(
    num_perm=128,
    seed=SEED,
    lsh_threshold=0.8,
    num_bands=16,
    final_jaccard_threshold=0.85,
)

document_dedup_stage = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_1),
        rensa_index,
        JsonlReader(output_intermediate_1),
        RensaDeduplicate(
            rensa_index=rensa_index,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/8_dedup/{CORPUS}")
        ),
        JsonlWriter(output_intermediate_2),
    ],
    logging_dir=f"{LOGGING_DIR}/minhash/{CORPUS}",
    depends=main_processing_executor,
)

reshard_stage = LocalPipelineExecutor(
    pipeline=[
        JsonlShard(
            input_folder=output_intermediate_2,
            output_folder=output_intermediate_3,
            num_shards=8,
        ),
    ],
    depends=document_dedup_stage,
)

final_stage = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_2),
        DocStats(
            f"{STATS_DIR}/docs",
            top_k_config=top_k_config,
        ),
        LineStats(
            f"{STATS_DIR}/lines",
            top_k_config=top_k_config,
        ),
        TokenStats(
            f"{STATS_DIR}/tokens",
            top_k_config=top_k_config,
        ),
        PIIFormatter(),
        JsonlWriter(f"{MAIN_OUTPUT_DIR}/deduped/{CORPUS}"),
    ],
    tasks=8,
    workers=8,
    logging_dir=f"{LOGGING_DIR}/final/{CORPUS}",
    depends=document_dedup_stage,
)



def main():
    main_processing_executor.run()
    document_dedup_stage.run()
    reshard_stage.run()
    final_stage.run()

if __name__ == "__main__":
    main()
