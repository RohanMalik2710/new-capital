---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:77
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: What fruit trees are suitable for hot climates?
  sentences:
  - The transcript mentions HRM 99 as a potential option for hot climates, but more
    details are needed for complete understanding.
  - For Kharif crops, fertilizer management needs to adapt to rainfall. Excessive
    rain can wash away fertilizers, while drought limits fertilizer use. In heavy
    rain, significant fertilizer loss can occur.  In drought conditions, urea-based
    fertilizers may be unsuitable.  An integrated approach is necessary, including
    using biofertilizers to enhance nutrient uptake and reduce reliance on chemical
    fertilizers. Biofertilizers help utilize existing soil nutrients, while nitrogen-fixing
    bacteria (like Azotobacter and Rhizobium) provide atmospheric nitrogen to the
    plants.
  - First, determine the level of alkalinity. If the pH is between 6.5 and 8, banana
    cultivation is possible.  Get your soil tested to determine the exact pH. If the
    pH is higher than 8, it's not suitable.  High alkalinity means increased water
    needs, and if water doesn't retain well in the field, it will leach out nutrients,
    causing problems.  So, get your soil tested; if it's highly alkaline, it's not
    suitable. If the pH is between 6.5 and 8, then cultivation is possible.
- source_sentence: Where can I obtain G9 banana saplings for cultivation?
  sentences:
  - During flowering, plants need more phosphorus.  Nano DAP can help supplement any
    deficiencies remaining after initial fertilizer application, leading to better
    results. If you haven't applied potash, you can use a 0-0-50 grade potash fertilizer
    (e.g., 20 grams per liter of water) as a spray to improve grain quality.
  - The provided text mentions the problem but doesn't offer a solution for excessive
    rain and fruit rot in tomatoes.
  - You can acquire G9 banana saplings from reputable institutions, good private nurseries,
    or agricultural science centers.  Collaborating with scientists from agricultural
    universities is also recommended.
- source_sentence: What is one advantage of banana cultivation?
  sentences:
  - Once planted, banana plants can produce fruit for at least 7 years.  The first
    harvest takes about 13 months, subsequent harvests occur every 8-9 months, and
    then 6-7 months after that.  However, proper management is crucial.
  - Banana cultivation is widespread in India, with major production in Maharashtra,
    Tamil Nadu, and increasingly, Uttar Pradesh.  It constitutes approximately 30%
    of total fruit production in the country.
  - Balanced fertilizer use means providing all essential nutrients in the right amounts
    and proportions at the right time, according to the crop's needs. Plants need
    18 nutrients for complete growth; nitrogen, phosphorus, and potassium are used
    in larger amounts, while others are needed in smaller quantities.  Balanced use
    ensures the crop receives these nutrients in the correct ratio.
- source_sentence: How to deal with pest and disease outbreaks in crops?
  sentences:
  - Immediately spray the affected area to prevent the disease from spreading further.
  - Send your questions via email to hello.kisan.2015@gmail.com or call the DD Kisan
    call center at [phone number redacted -  unclear from the provided text].
  - Bananas are rich in carbohydrates, contain some protein,  have a low fat content,
    are a good source of various minerals and vitamins, and are particularly rich
    in potassium.
- source_sentence: How can farmers manage fertilizers efficiently and economically?
  sentences:
  - Remove unwanted suckers.  Maintain proper drainage by mounding soil around the
    plants. Provide support (staking) for fruiting plants.
  - Use potash fertilizer. It strengthens the crop and increases its resistance to
    diseases and pests.
  - Use an integrated management system. Compost old crops to use as fertilizer. Utilize
    biofertilizers with every crop.  Use recommended amounts of urea and DAP, avoiding
    excess. Explore alternatives like nano-urea and liquid fertilizers to reduce chemical
    fertilizer use and protect the soil and environment.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
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
    'How can farmers manage fertilizers efficiently and economically?',
    'Use an integrated management system. Compost old crops to use as fertilizer. Utilize biofertilizers with every crop.  Use recommended amounts of urea and DAP, avoiding excess. Explore alternatives like nano-urea and liquid fertilizers to reduce chemical fertilizer use and protect the soil and environment.',
    'Use potash fertilizer. It strengthens the crop and increases its resistance to diseases and pests.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5685, 0.4091],
#         [0.5685, 1.0000, 0.6366],
#         [0.4091, 0.6366, 1.0000]])
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

* Size: 77 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 77 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 10 tokens</li><li>mean: 17.05 tokens</li><li>max: 29 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 48.56 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                        | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
  |:------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What is the problem of Panama wilt in banana plants and how can we solve it?</code>                         | <code>Panama wilt is a disease that affects banana plants.  The source of the planting material and the age of the plants are important factors. Older plants are more susceptible. Crop rotation is recommended if Panama wilt is a recurring problem in the field.  Changing the plant variety may also be necessary.</code>                                                                                                                                                                                                                        |
  | <code>How often should soil testing be done, and what is the proper procedure for collecting soil samples?</code> | <code>Soil testing should be done every 2-3 years. For a 1-acre field, collect samples from 78 different locations.  Dig a V-shaped hole 15 centimeters deep (about 2.5 inches) at each location. Take a soil sample from this layer. Mix all samples together and take a 400-gram subsample. Send this to a reputable soil testing laboratory, such as those associated with IFFCO (Indian Farmers Fertiliser Cooperative). The lab will analyze the soil's nutrient levels and recommend appropriate fertilizers based on your planned crop.</code> |
  | <code>How are secondary nutrients like magnesium, sulfur, and calcium used in fertilizer?</code>                  | <code>Secondary nutrients like magnesium, sulfur, and calcium are recommended for crops with higher needs for these elements. For example, sulfur is crucial for crops like garlic, onions, and peanuts (oilseed crops).</code>                                                                                                                                                                                                                                                                                                                       |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.12.7
- Sentence Transformers: 5.0.0
- Transformers: 4.45.2
- PyTorch: 2.5.0+cpu
- Accelerate: 1.9.0
- Datasets: 4.0.0
- Tokenizers: 0.20.1

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
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
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