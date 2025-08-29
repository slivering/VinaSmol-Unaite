

patch = """
vinasmol = dict(
    name="VinaSmol-360M",
    hf_config=dict(org="HuggingFaceTB", name="SmolLM2-360M-Instruct"), # TODO
    block_size=8192,
    vocab_size=49152, # TODO
    padded_vocab_size=49152, # TODO
    n_layer=32,
    n_head=15,
    n_embd=960,
    n_query_groups=5,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    norm_class_name="RMSNorm",
    mlp_class_name="LLaMAMLP",
    intermediate_size=2560,
    rope_base=100000,
    norm_eps=1e-5,
)

configs.append(vinasmol)

name_to_config = {config["name"]: config for config in configs}
"""

# TODO: patch litgpt/config.py