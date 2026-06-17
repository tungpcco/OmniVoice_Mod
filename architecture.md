# OmniVoice Project Architecture

This document maps the project directory structure and explains the purpose of each component in the **OmniVoice** repository.

---

## Directory Tree

```
OmniVoice_src/
├── .github/                     # GitHub configuration (Issue templates)
├── docs/                        # Project documentation files (Markdown & IPython Notebook)
├── examples/                    # Example training configs and shell scripts
│   └── config/                  # Hyperparameter config files (JSON)
├── omnivoice/                   # Main python package containing all source code
│   ├── cli/                     # Command-line interface entrypoints
│   ├── data/                    # Dataset parsing, tokenization, batching, collation
│   ├── eval/                    # Evaluation code (WER, UTMOS, Speaker Similarity)
│   │   ├── models/              # Evaluation neural network modules
│   │   ├── mos/                 # Mean Opinion Score computation
│   │   ├── speaker_similarity/  # Speaker similarity evaluation
│   │   └── wer/                 # Word Error Rate computation scripts
│   ├── models/                  # Core model architecture (diffusion language model)
│   ├── scripts/                 # Utility scripts (denoising, audio token extraction)
│   ├── training/                # Training modules (trainer, building optimizer/scheduler)
│   └── utils/                   # Shared utility functions (audio, text, devicetypes)
├── LICENSE                      # Apache-2.0 License
├── pyproject.toml               # Python project configuration and dependency specifications
├── README.md                    # Main repository README documentation
└── uv.lock                      # uv package manager lockfile
```

---

## Detailed Component Map

### 1. `omnivoice/cli/` (Command-Line Interface)
Entrypoint scripts for running the application in various modes.
- **[demo.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/cli/demo.py)**: Gradio web interface demo script. Permits interactive zero-shot text-to-speech synthesis.
- **[infer.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/cli/infer.py)**: Single utterance command-line inference script.
- **[infer_batch.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/cli/infer_batch.py)**: Performs batch inference over metadata files.
- **[train.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/cli/train.py)**: Entrypoint for training/fine-tuning.

### 2. `omnivoice/data/` (Data Processing)
Data loaders, tokenization, batching structures.
- **[batching.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/data/batching.py)**: Implements dynamic batching based on token length to maximize GPU utilization.
- **[collator.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/data/collator.py)**: Custom collator to pad and batch text, phonemes, and audio tokens.
- **[dataset.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/data/dataset.py)**: Main Dataset classes for loading audio-text pairs, supporting WebDataset.
- **[processor.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/data/processor.py)**: Handles text normalization, language identification, and speech tokenizer conversion.

### 3. `omnivoice/eval/` (Evaluation Suite)
Scripts and models to evaluate synthesized speech.
- **`models/`**: Includes [ecapa_tdnn_wavlm.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/eval/models/ecapa_tdnn_wavlm.py) (speaker similarity encoder) and [utmos.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/eval/models/utmos.py) (MOS predictor).
- **`mos/`**: Evaluates Mean Opinion Score using UTMOS.
- **`speaker_similarity/`**: Measures Cosine Similarity between speaker embedding vectors of source and synthesized voices.
- **`wer/`**: Word Error Rate computation scripts using various speech recognition models (SenseVoice, Hubert, etc.).

### 4. `omnivoice/models/` (Core Architecture)
Neural network model specifications.
- **[omnivoice.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/models/omnivoice.py)**: The core model architecture of OmniVoice, implementing a diffusion-based language model for multilingual zero-shot text-to-speech.

### 5. `omnivoice/scripts/` (Preprocessing & Utilities)
Standalone preprocessing pipelines.
- **[denoise_audio.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/scripts/denoise_audio.py)**: Pre-processing step to clean noisy audio datasets.
- **[extract_audio_tokens.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/scripts/extract_audio_tokens.py)**: Extracts discrete speech tokens using pre-trained audio encoders.
- **[jsonl_to_webdataset.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/scripts/jsonl_to_webdataset.py)**: Converts JSONL index metadata into WebDataset tar format for efficient loading.

### 6. `omnivoice/training/` (Training Framework)
Classes and loops for model training.
- **[trainer.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/training/trainer.py)**: PyTorch/Accelerate trainer coordinating multi-GPU/distributed training.
- **[builder.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/training/builder.py)**: Factory functions for models, optimizers, learning rate schedulers, and data loaders.
- **[checkpoint.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/training/checkpoint.py)**: Handles loading, saving, and pruning of model states.

### 7. `omnivoice/utils/` (Common Utilities)
- **[common.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/utils/common.py)**: Device auto-detection (CUDA, XPU, MPS, CPU) and seed settings.
- **[audio.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/utils/audio.py)**: Sampling, normalisation, and format conversions.
- **[text.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/utils/text.py)**: Phonemizer helpers, cleanups, and formatting.
- **[lang_map.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/utils/lang_map.py)**: Multilingual maps and metadata dictionary mapping languages.
- **[voice_design.py](file:///d:/AI_DATA/OmniVoice/OmniVoice_src/omnivoice/utils/voice_design.py)**: Search logic to find/select matching speaker prompt audio files based on characteristics.
