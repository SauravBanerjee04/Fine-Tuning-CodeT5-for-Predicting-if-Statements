# Fine-Tuning CodeT5 for Predicting If Statements

## Overview
This project fine-tunes the CodeT5 model (`Salesforce/codet5-small`) to predict masked `if` conditions in Python functions. It covers data preprocessing, model training, evaluation, and result reporting—all executed within Google Colab for easy replication.

## Repository Structure
All files live at the repository root:
```
├── ft_train.csv                   # Training dataset (50,000 samples)
├── ft_valid.csv                   # Validation dataset (5,000 samples)
├── ft_test.csv                    # Test dataset (5,000 samples)
├── FineTune_CodeT5.ipynb          # Jupyter notebook with all steps
├── Fine-Tune_CodeT5 Lab Description.pdf  # Assignment description
├── outputs/                       # Directory for model checkpoints and results
│   ├── codet5_finetuned_gpu/      # Saved model and checkpoints
│   ├── testset-results.csv        # Test predictions & metrics CSV
│   └── metrics_summary.txt        # Summary of evaluation metrics
└── README.md                      # This file
```

## Running in Google Colab
1. **Open in Colab**: In GitHub, navigate to `FineTune_CodeT5.ipynb` and click **Open in Colab**.
2. **Select GPU runtime**: In Colab, go to **Runtime > Change runtime type** and choose **GPU**.
3. **Clone the repository**:
   ```bash
   !git clone <your-repo-url>
   %cd <your-repo-directory>
   ```
4. **Upload data**: Ensure `ft_train.csv`, `ft_valid.csv`, and `ft_test.csv` are in the notebook’s working directory (either by cloning or uploading via the Colab UI).
5. **Run all cells**: Use **Runtime > Run all** to execute the notebook end-to-end.

## Step-by-Step Notebook Explanation

### Cell 1: Install Dependencies & Download NLTK Data
- Installs specific versions of PyTorch (with CUDA 12.4 support), Hugging Face Transformers, Datasets, Evaluate, Tree-sitter, SacreBLEU, Weights & Biases, and NLTK.
- Downloads the NLTK `punkt` and `punkt_tab` tokenizers for sentence splitting.

### Cell 2: Load Pre-trained Model and Tokenizer
- Specifies the CodeT5 small model (`Salesforce/codet5-small`).
- Loads the corresponding tokenizer and model from Hugging Face’s model hub.
- Verifies successful loading with a print statement.

### Cell 3: Masking and Flattening Functions
- Defines `mask_if_condition()`:
  - Flattens multi-line function code into a single line.
  - Finds the first `if` condition via regex and replaces its condition with `<mask>`.
- Demonstrates the function on an example snippet.
- Applies the masking function to the `cleaned_method` column of the training, validation, and test CSVs, creating a new `masked_function` column in each DataFrame.

### Cell 4: Dataset Class & DataLoader Setup
- Defines `FineTuneDataset`, a subclass of `torch.utils.data.Dataset`:
  - Takes a DataFrame, tokenizer, and max lengths for inputs/targets.
  - Returns tokenized inputs and labels ready for training.
- Instantiates the dataset and a `DataLoader` for the training split (batch size 8) and prints the shapes of one batch for verification.

### Cell 5: Training Configuration & Execution
- **GPU Memory Management**:
  - Sets `PYTORCH_CUDA_ALLOC_CONF` to reduce fragmentation.
  - Enables gradient checkpointing to save memory.
- **Callbacks**:
  - `ClearCacheCallback` empties GPU cache at each epoch end.
  - `ClearEvaluationCallback` triggers CPU garbage collection after evaluation.
- **TrainingArguments**:
  - Defines output directory, evaluation and saving strategies, learning rate, batch sizes (64), number of epochs (5), FP16 mixed precision, early stopping (patience 3), and logging.
- Initializes the `Trainer` with model, datasets, tokenizer, metrics, and callbacks.
- Calls `trainer.train()` to start fine-tuning; best model and checkpoints are saved under `outputs/codet5_finetuned_gpu/`.

### Cell 6: Generating Predictions & Computing Fine-Grained Metrics
- Reloads the test CSV and wraps it in a `FineTuneDataset` and `DataLoader`.
- Puts the model in evaluation mode and generates predictions with beam search (5 beams).
- Aligns each predicted token sequence to its reference length using the tokenizer’s pad token.
- Computes per-example BLEU-4 (via NLTK smoothing) and sentence-level CodeBLEU using `evaluate`’s SacreBLEU metric.
- Constructs a DataFrame of inputs, references, predictions, exact-match flags, BLEU, and CodeBLEU scores.
- Saves this to `outputs/testset-results.csv`.

### Cell 7: Summary Metrics & Reporting
- Loads `testset-results.csv` into a DataFrame.
- Computes:
  - **Exact Match Rate**
  - **Average BLEU-4**
  - **Corpus-level SacreBLEU**
  - **Average Sentence-level CodeBLEU**
  - **Token-level F1, Precision, Recall**
  - **Average Length Difference** between predictions and references
- Prints each metric and writes them to `outputs/metrics_summary.txt`.

## Reproducing the Workflow Automatically
In Colab, after cloning the repo, you can execute the notebook headlessly:
```bash
!jupyter nbconvert --execute --to notebook --inplace FineTune_CodeT5.ipynb
```

## Tips
- If you run out of GPU memory, reduce `per_device_train_batch_size` in Cell 5.
- To run without GPU, disable `fp16=True` and lower batch sizes.
- Feel free to experiment with preserving indentation tokens (e.g., `<TAB>`) in masking for improved structure awareness.


