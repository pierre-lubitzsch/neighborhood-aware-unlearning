# Neighborhood-Aware Unlearning for Generative Recommendation

Pierre Lubitzsch, Yubao Tang, Maarten de Rijke, Sebastian Schelter

[Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX)

This repository contains the code for Neighborhood-Aware Unlearning (NAU), a method for removing training interactions from generative recommenders that predict items through semantic IDs (SIDs). It builds on [GRID](https://github.com/snap-research/GRID) (Generative Recommendation with Semantic IDs). We keep GRID's data format, semantic-ID pipeline, and TIGER implementation, and add:

- two deletion scenarios:
  - **spam removal:** a bandwagon attack injects fake users that promote a target item;
  - **unwanted-item removal:** users request deletion of their interactions with an item category;
- the retrained references for both scenarios;
- sequential unlearning with NAU and the baselines Finetune, Forget only, Forget+Repair, SCIF, SEIF, Kookmin, Fanchuan, TRACER, and Filter (SCIF, SEIF, Kookmin, and Fanchuan are ported from the [ERASE benchmark](https://github.com/deem-data/erase-bench) [6]);
- LETTER semantic IDs, in addition to GRID's RQ-KMeans and RQ-VAE;
- evaluation of exposure metrics (SH@10, UHF@10, UHR@10) next to NDCG@10 and Recall@10.

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup Environment

```bash
git clone https://github.com/pierre-lubitzsch/neighborhood-aware-unlearning.git
cd neighborhood-aware-unlearning
pip install -r requirements.txt
```

All commands are run from the repository root.

## 🎯 Data Preparation

Datasets are expected in the GRID format:
```
src/data/amazon_data/<dataset>/
├── training/    # training sequences of user histories
├── evaluation/  # validation sequences of user histories
├── testing/     # test sequences of user histories
└── items/       # text of all items in the dataset
```

We use the Amazon Beauty, Sports, and Toys datasets preprocessed as in the P5 paper [4]. GRID provides them in this format; see the download link in the [GRID README](https://github.com/snap-research/GRID#1-data-preparation). The datasets are not redistributed here.

## Base GRID workflow

These are GRID's commands; the pipeline scripts below wrap them.

Embeddings of the item texts:
```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=src/data/amazon_data/beauty
```

Semantic IDs (3 codebooks of 256 centroids on 2048-dimensional flan-t5-xl embeddings):
```bash
python -m src.train experiment=rkmeans_train_flat data_dir=src/data/amazon_data/beauty \
    embedding_path=<embeddings>.pt embedding_dim=2048 num_hierarchies=3 codebook_width=256
python -m src.inference experiment=rkmeans_inference_flat data_dir=src/data/amazon_data/beauty \
    embedding_path=<embeddings>.pt embedding_dim=2048 num_hierarchies=3 codebook_width=256 ckpt_path=<codebook checkpoint>
```

Generative recommender. `num_hierarchies=4`, because a digit is appended to de-duplicate the semantic IDs:
```bash
python -m src.train experiment=tiger_train_flat data_dir=src/data/amazon_data/beauty \
    semantic_id_path=<semantic ids>.pt num_hierarchies=4
```

## Unlearning pipeline

`scripts/pipeline/` contains one script per step. Each script documents its arguments in its header, and the scripts that train or unlearn accept extra Hydra overrides as trailing arguments. The example uses Beauty; replace `beauty` by `sports` or `toys` for the other datasets.

```bash
D=src/data/amazon_data/beauty
mkdir -p outputs
```

### 1. Item embeddings and semantic IDs

```bash
bash scripts/pipeline/embed_items.sh $D outputs/beauty_emb.pt
bash scripts/pipeline/train_sid.sh   $D outputs/beauty_emb.pt outputs/beauty_sid.pt        # RQ-KMeans

QUANTIZER=rqvae bash scripts/pipeline/train_sid.sh $D outputs/beauty_emb.pt outputs/beauty_sid_rqvae.pt
bash scripts/pipeline/letter_sid.sh $D outputs/beauty_emb.pt outputs/beauty_sid_letter.pt   # LETTER
```

`train_sid.sh` prints the path of the quantizer checkpoint, which TRACER needs (step 5). `letter_sid.sh` first trains collaborative-filtering item embeddings (SASRec), then the LETTER tokenizer, and finally assigns collision-free IDs.

### 2a. Spam scenario (poisoning)

A 1% bandwagon attack promoting one target item. The target is drawn from the `unpopular` (bottom 20% by interaction count), `mid` (30 to 70%), or `popular` (top 5%) items.

```bash
# <clean dir> <out dir> <strategy> <seed>
bash scripts/pipeline/poison.sh $D outputs/beauty_spam_mid mid 2
```

The output directory contains:
- the poisoned `training/` data;
- the forget set `training_forget/` (the fake users);
- the retain set `training_retain/`;
- `forget_manifest.json`.

### 2b. Unwanted-item scenario (selection of sensitive items)

The unwanted items form a category, given either as a taxonomy node of the item metadata or as title keywords. Requesting users are sampled among those who interacted with the category until the requested share of interactions is reached (`FORGET_RATIO`, default `1e-4`).

```bash
# Beauty: Hair Loss Products (taxonomy level 2); Sports: "Guns & Rifles" (level 3)
CATEGORY="Hair Loss Products" LEVEL=2 bash scripts/pipeline/select_unwanted.sh $D outputs/beauty_hairloss 2

# Toys: toy weapons, by keyword
KEYWORDS="gun guns rifle rifles pistol pistols weapon weapons sword swords knife knives dagger" \
  bash scripts/pipeline/select_unwanted.sh src/data/amazon_data/toys outputs/toys_weapons 2
```

This writes two directories:
- `outputs/beauty_hairloss_sens/`: the forget/retain split of the requested (user, item) interactions. Its evaluation data and items link to the clean dataset.
- `outputs/beauty_hairloss_retrain/`: the training data without those interactions.

### 3. Training and retraining

The same script trains the original model and the retrained reference; only the data directory differs:

| Scenario | Original model | Retrained reference |
|---|---|---|
| spam | poisoned dir | clean dir |
| unwanted items | clean dir | `*_retrain` dir |

```bash
SID=outputs/beauty_sid.pt
bash scripts/pipeline/train_rec.sh outputs/beauty_spam_mid $SID outputs/rec_poisoned 2          # <data> <sids> <run dir> <seed>
bash scripts/pipeline/train_rec.sh $D $SID outputs/rec_clean 2
bash scripts/pipeline/train_rec.sh outputs/beauty_hairloss_retrain $SID outputs/rec_retrain_hairloss 2
```

Checkpoints are written to `<run dir>/checkpoints/`. For LETTER, set `MODEL=letter` and pass the LETTER semantic IDs.

### 4. Evaluation

```bash
# <checkpoint> <clean dir> <sids> <out dir> [forget manifest] [seed]
bash scripts/pipeline/evaluate.sh outputs/rec_clean/checkpoints/last.ckpt $D $SID outputs/eval_retrained \
  outputs/beauty_spam_mid/forget_manifest.json
```

`<out dir>/csv/version_0/metrics.csv` contains:
- `test/ndcg@10`, `test/recall@10`: recommendation utility;
- `test/SH@10`: exposure of the promoted item (spam);
- `test/SHF@10`, `test/SHR@10`: unwanted-item hit rate of the requesting and of the remaining users (UHF@10 and UHR@10 in the paper).

The distances to retraining, Δ_SH = |SH_U − SH_R| and Δ_UHR, are computed from these values against the retrained model of the same seed. So is utility retention, τ_P = NDCG_U / NDCG_R.

### 5. Unlearning

```bash
# spam removal with NAU
bash scripts/pipeline/unlearn.sh nau outputs/rec_poisoned/checkpoints/last.ckpt \
  outputs/beauty_spam_mid $D $SID outputs/beauty_emb.pt outputs/ul_spam_nau 2

# unwanted-item removal with NAU
bash scripts/pipeline/unlearn.sh nau_unwanted outputs/rec_clean/checkpoints/last.ckpt \
  outputs/beauty_hairloss_sens $D $SID outputs/beauty_emb.pt outputs/ul_hairloss_nau 2
```

Arguments: `<method> <checkpoint> <dir with forget/retain split> <clean dir> <sids> <item embeddings> <run dir> [seed] [overrides...]`.

Deletion requests are processed sequentially. The unlearned model is saved to `<run dir>/checkpoints/unlearned.ckpt` and evaluated on the test split into `<run dir>/eval/`.

| Method | Settings |
|---|---|
| `nau` | λ_f=0.1, λ_s=0.01, λ_n=0.01 (spam removal) |
| `nau_unwanted` | λ_f=0.1, λ_s=0, λ_n=−1 (unwanted-item removal) |
| `finetune` | fine-tuning on the retain set |
| `forget_only` | gradient ascent on the forget set, λ_f=1 |
| `forget_repair` | forget term λ_f=0.1 with retain-set repair |
| `scif` | SCIF |
| `seif` | SEIF, noise std 0.06 |
| `kookmin` | Kookmin, init rate 0.001 |
| `fanchuan` | Fanchuan, temperature 0.07 |
| `tracer` | TRACER, λ_forget=1, λ_coherence=0.1; append `unlearning.tracer_codebook_ckpt=<quantizer checkpoint>` |
| `filter` | decoding-time filter of the forgotten items |

NAU's objective weights are `unlearning.lambda_f` (forget), `unlearning.lambda_s` (separation), and `unlearning.lambda_n` (neighborhood). A negative λ_n suppresses probability mass on the neighborhood of the forgotten items instead of redistributing it there. Weights are overridden by appending them, for example:

```bash
bash scripts/pipeline/unlearn.sh nau <checkpoint> outputs/beauty_spam_mid $D $SID outputs/beauty_emb.pt outputs/ul_nau_ln0.1 2 \
  unlearning.lambda_n=0.1
```

Baseline hyperparameters are overridden the same way:
- `unlearning.seif_erase_std`
- `unlearning.kookmin_init_rate`
- `unlearning.fanchuan_contrastive_temperature`
- `unlearning.tracer_lambda_forget`, `unlearning.tracer_lambda_coherence`

All options and their defaults are in `configs/experiment/tiger_unlearn_scif_flat.yaml` and `configs/unlearning_defaults.yaml`.

### Validation split

For hyperparameter selection on the validation split, create a view of the dataset whose `testing/` points to `evaluation/`. Pass it as the clean data directory of `evaluate.sh` and `unlearn.sh`:

```bash
V=src/data/amazon_data/beauty_valview
mkdir -p $V
for s in training items evaluation; do ln -s "$(readlink -f $D/$s)" $V/$s; done
ln -s "$(readlink -f $D/evaluation)" $V/testing
```

## Experiment grid of the paper

The paper's experiments repeat the pipeline over the following grid:

- **Datasets:** Beauty, Sports, Toys.
- **Data seeds:** 2, 3, 5, 7, 11. The seed selects the attack or the requesting users; the recommender is trained with the same seed.
- **Spam targets:** `unpopular`, `mid`, `popular` (step 2a), one target item per poisoned dataset.
- **Unwanted categories** (step 2b): Hair Loss Products (Beauty), Guns & Rifles (Sports), toy weapons (Toys).
- **Semantic IDs:** RQ-KMeans for the main results; RQ-VAE and LETTER as additional tokenizers.

Every unlearned model is compared with the retrained model of the same dataset, seed, and target or category (step 3).

Training uses all visible GPUs (`trainer.devices=-1`). Unlearning runs on a single device.

## Repository layout

- `configs/`: Hydra configurations; each `experiment/*_flat.yaml` is a complete setup.
- `src/models/`: TIGER, LETTER, and the unlearning module.
- `src/components/unlearning/`: NAU and the baseline algorithms.
- `src/data/poisoning/`: bandwagon attack generation.
- `src/data/unlearning/`: forget/retain splits and deletion specifications.
- `scripts/`: sensitive-item selection, retrain datasets, LETTER IDs, evaluation, and `pipeline/`.

## Supported Models

### Semantic ID
1. Residual K-means proposed in OneRec [2]
2. Residual Quantization with Variational Autoencoder [3]
3. LETTER [5]

### Generative Recommendation
1. TIGER [1]

## 📚 Citation

If you use this code, please cite our paper:

```bibtex
@misc{lubitzsch2026nau,
  title         = {Neighborhood-Aware Unlearning for Generative Recommendation},
  author        = {Lubitzsch, Pierre and Tang, Yubao and de Rijke, Maarten and Schelter, Sebastian},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR}
}
```

This code builds on GRID:

```bibtex
@inproceedings{grid,
  title     = {Generative Recommendation with Semantic IDs: A Practitioner's Handbook},
  author    = {Ju, Clark Mingxuan and Collins, Liam and Neves, Leonardo and Kumar, Bhuvesh and Wang, Louis Yufeng and Zhao, Tong and Shah, Neil},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```

## 🤝 Acknowledgments

- Built on [GRID](https://github.com/snap-research/GRID), which is built on top of https://github.com/ashleve/lightning-hydra-template
- [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/), and [Hydra](https://hydra.cc/)

## Bibliography

[1] Rajput, Shashank, et al. "Recommender systems with generative retrieval." Advances in Neural Information Processing Systems 36 (2023): 10299-10315.

[2] Deng, Jiaxin, et al. "OneRec: Unifying retrieve and rank with generative recommender and iterative preference alignment." arXiv preprint arXiv:2502.18965 (2025).

[3] Lee, Doyup, et al. "Autoregressive image generation using residual quantization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[4] Geng, Shijie, et al. "Recommendation as language processing (RLP): A unified pretrain, personalized prompt & predict paradigm (P5)." Proceedings of the 16th ACM Conference on Recommender Systems. 2022.

[5] Wang, Wenjie, et al. "Learnable item tokenization for generative recommendation." Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2024.

[6] Lubitzsch, Pierre Sicco, Maarten de Rijke, and Sebastian Schelter. "ERASE: A Real-World Aligned Benchmark for Unlearning in Recommender Systems." Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2026.

## License

This code is derived from GRID and is distributed under GRID's license (`LICENSE`): non-commercial research use only. Third-party attribution notices are in `notices.txt`.
