---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:1940
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: 'Follow order of operations to evaluate: 16 - 2(5 + 1) + 20 / 4'
  sentences:
  - 'Recognize Quadratic Pattern (Algebra & Functions): Sequences with constant second
    differences, sequence patterns with differences increasing or decreasing by a
    constant amount each term, and quadratic formulas.'
  - 'Order of Operations All (Number Sense & Operations): Multi-step numerical evaluation
    and arithmetic calculations following PEMDAS and BODMAS order of operations rules
    with parentheses, division, multiplication, addition, and subtraction.'
  - 'Order of Operations All (Number Sense & Operations): Multi-step numerical evaluation
    and arithmetic calculations following PEMDAS and BODMAS order of operations rules
    with parentheses, division, multiplication, addition, and subtraction.'
- source_sentence: 'Evaluate the expression using BODMAS: 18 - 2(3 + 5) + 15 / 3'
  sentences:
  - 'Order of Operations All (Number Sense & Operations): Multi-step numerical evaluation
    and arithmetic calculations following PEMDAS and BODMAS order of operations rules
    with parentheses, division, multiplication, addition, and subtraction.'
  - 'Scale Factor (Proportional Reasoning & Percents): Map scales, blueprint measurements,
    ratio multipliers, scale drawings, and distance representations between map and
    reality.'
  - 'Equation Solving Two or Fewer Steps (Algebra & Functions): Solving linear equations,
    finding an unknown mystery number from verbal statements or operations, and isolating
    variables in one or two steps.'
- source_sentence: 'Evaluate: 12 + 4(5 - 2) - 18 / 3'
  sentences:
  - 'Recognize Quadratic Pattern (Algebra & Functions): Sequences with constant second
    differences, sequence patterns with differences increasing or decreasing by a
    constant amount each term, and quadratic formulas.'
  - 'Equation Solving Two or Fewer Steps (Algebra & Functions): Solving linear equations,
    finding an unknown mystery number from verbal statements or operations, and isolating
    variables in one or two steps.'
  - 'Order of Operations All (Number Sense & Operations): Multi-step numerical evaluation
    and arithmetic calculations following PEMDAS and BODMAS order of operations rules
    with parentheses, division, multiplication, addition, and subtraction.'
- source_sentence: What is the expanded form of 5(a + 3b)?
  sentences:
  - 'Distributive Property (Algebra & Functions): ASSISTments skill 340: Distributive
    Property in Algebra & Functions.'
  - 'Algebraic Solving (Algebra & Functions): ASSISTments skill 166: Algebraic Solving
    in Algebra & Functions.'
  - 'D.4.8-understanding-concept-of-probabilities (Statistics & Probability): ASSISTments
    skill 110: D.4.8-understanding-concept-of-probabilities in Statistics & Probability.'
- source_sentence: Calculate the value of 15 - 2(3 + 4) + 16 / 4
  sentences:
  - 'Order of Operations All (Number Sense & Operations): Multi-step numerical evaluation
    and arithmetic calculations following PEMDAS and BODMAS order of operations rules
    with parentheses, division, multiplication, addition, and subtraction.'
  - 'Order of Operations All (Number Sense & Operations): Multi-step numerical evaluation
    and arithmetic calculations following PEMDAS and BODMAS order of operations rules
    with parentheses, division, multiplication, addition, and subtraction.'
  - 'Recognize Quadratic Pattern (Algebra & Functions): Sequences with constant second
    differences, sequence patterns with differences increasing or decreasing by a
    constant amount each term, and quadratic formulas.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 64 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Calculate the value of 15 - 2(3 + 4) + 16 / 4',
    'Order of Operations All (Number Sense & Operations): Multi-step numerical evaluation and arithmetic calculations following PEMDAS and BODMAS order of operations rules with parentheses, division, multiplication, addition, and subtraction.',
    'Recognize Quadratic Pattern (Algebra & Functions): Sequences with constant second differences, sequence patterns with differences increasing or decreasing by a constant amount each term, and quadratic formulas.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5869, 0.0254],
#         [0.5869, 1.0000, 0.0387],
#         [0.0254, 0.0387, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,940 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                        | sentence_1                                                                         |
  |:---------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                             |
  | modality | text                                                                              | text                                                                               |
  | details  | <ul><li>min: 8 tokens</li><li>mean: 23.96 tokens</li><li>max: 61 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 35.62 tokens</li><li>max: 50 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                      | sentence_1                                                                                                                                                                                                                                                 |
  |:----------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Evaluate: 18 - 3(4 + 2) + 24 / 6</code>                                                                   | <code>Order of Operations All (Number Sense & Operations): Multi-step numerical evaluation and arithmetic calculations following PEMDAS and BODMAS order of operations rules with parentheses, division, multiplication, addition, and subtraction.</code> |
  | <code>A model car has a scale factor of 1:24. If the model is 18 cm long, what is the actual car length?</code> | <code>Scale Factor (Proportional Reasoning & Percents): Map scales, blueprint measurements, ratio multipliers, scale drawings, and distance representations between map and reality.</code>                                                                |
  | <code>Identify the quadratic rule for sequence 6, 13, 22, 33 where differences are 7, 9, 11.</code>             | <code>Recognize Quadratic Pattern (Algebra & Functions): Sequences with constant second differences, sequence patterns with differences increasing or decreasing by a constant amount each term, and quadratic formulas.</code>                            |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Time
- **Training**: 12.1 minutes

### Framework Versions
- Python: 3.11.0
- Sentence Transformers: 5.7.0
- Transformers: 5.15.0
- PyTorch: 2.13.0+cpu
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->