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

from . import (
    DATA_DIR_CODE, MAIN_OUTPUT_DIR, FILTERING_OUTPUT_DIR, FILTERING_REMOVED_DIR,
    ES_DIR, LOGGING_DIR, STATS_DIR, SEED,
)
CORPUS = "code-all"

output_intermediate_1 = f"{FILTERING_OUTPUT_DIR}/output_1/{CORPUS}"
output_intermediate_2 = f"{FILTERING_OUTPUT_DIR}/output_2/{CORPUS}"
es_dir_en = f"{ES_DIR}/{CORPUS}"

top_k_config = TopKConfig(top_k_groups=[])

main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            str(DATA_DIR_CODE),
            recursive=True,
            limit=100, # TODO: remove, for debugging
            default_metadata={"dump": CORPUS},
        ),
        JsonlWriter(output_intermediate_1),
    ],
    tasks=16,
    workers=16,
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

tasks_sequence_dedup = 16

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
        JsonlWriter(f"{MAIN_OUTPUT_DIR}/{CORPUS}/deduped"),
    ],
    tasks=tasks_sequence_dedup,
    workers=tasks_sequence_dedup,
    logging_dir=f"{LOGGING_DIR}/es/3/{CORPUS}",
    depends=document_dedup_stage,
)



def main():
    main_processing_executor.run()
    document_dedup_stage.run()
    final_stage.run()

if __name__ == "__main__":
    main()
