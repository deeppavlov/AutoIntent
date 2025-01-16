"""Sample experiment."""

from pathlib import Path

from autointent import Dataset, Pipeline
from autointent.configs import EmbedderConfig, LoggingConfig

search_space = [
    {
        "node_type": "embedding",
        "metric": "retrieval_hit_rate",
        "search_space": [
            {
                "module_name": "retrieval",
                "k": [10],
                "embedder_name": ["avsolatorio/GIST-small-Embedding-v0", "infgrad/stella-base-en-v2"],
            }
        ],
    },
    {
        "node_type": "scoring",
        "metric": "scoring_roc_auc",
        "search_space": [
            {
                "module_name": "knn",
                "k": [1, 3, 5, 10],
                "weights": ["uniform", "distance", "closest"],
            },
            {"module_name": "linear"},
            {
                "module_name": "dnnc",
                "cross_encoder_name": [
                    "BAAI/bge-reranker-base",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                ],
                "k": [1, 3, 5],
            },
        ],
    },
    {
        "node_type": "decision",
        "metric": "decision_accuracy",
        "search_space": [
            {"module_name": "threshold", "thresh": [0.5]},
            {"module_name": "argmax"},
        ],
    },
]

log_config = LoggingConfig(
    report_to=["tensorboard"],
    dirpath=Path("clinc150_retrieval_hitrate_wandb"),
    dump_modules=False,
    run_name="bug_with_memory",
    clear_ram=True,
)
emb_config = EmbedderConfig(batch_size=16, device="cuda")

dataset = Dataset.from_hub("AutoIntent/clinc150_aug_qwen2.5-7b-awq")
pipeline_optimizer = Pipeline.from_search_space(search_space)
pipeline_optimizer.set_config(log_config)
pipeline_optimizer.set_config(emb_config)
pipeline_optimizer.fit(dataset)
