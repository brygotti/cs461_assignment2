# Report for Graded Assignment 2 - Part 2
> CS-461 Foundation Models and Generative AI

## Methods
The provided baseline (`linear_baseline`) first computes an unweighted average of the embeddings of each patch before passing them through a linear layer to obtain class logits. To improve upon this, we implement two attention-based methods inspired by the paper "Attention-based Deep Multiple Instance Learning" (Ilse et al., 2018). The main idea is to allow the model to attend differently to each patch of a given sample. In other words, instead of doing a simple average of the patch embeddings, we compute a weighted average where the weights are learned through an attention mechanism.

The first method (`attention`) is a direct implementation of the attention mechanism described in the paper. It uses a two-layer neural network with tanh activation function to compute attention scores for each patch embedding. These scores are then normalized using a softmax function to obtain attention weights, which are used to compute a weighted average of the patch embeddings. This weighted average is then passed through a linear layer to obtain class logits.

The second method (`attention_multi_head`) extends the first by incorporating multi-head attention. Instead of computing a single set of attention weights, we compute one set of attention weights for each class. This should in theory increase the model's expressivity by allowing each class logit to attend to different patches in the input sample. We modify the two-layer neural network so that its output is of size `num_classes` instead of 1, and we apply the softmax function across the patches for each class separately. We then compute a weighted average of the patch embeddings for each class using the corresponding attention weights, resulting in class-specific representations. Each representation is then passed through its own linear layer to obtain the final class logits. Using different linear layers for each class is important here, so that our model stays as expressive as the single-head attention model, where each class logit also has its own set of linear weights.

Both models are trained using the same 5-fold cross-validation setup as the baseline, with early stopping based on validation loss to prevent overfitting. Hyperparameters such as learning rate and batch size are kept the same as the baseline. To allow batch training, we implement a custom collate function that pads the number of patches in each sample to the maximum number of patches in the batch. A mask is also created to indicate which patches are valid and which are padding, ensuring that the attention mechanism only considers valid patches when computing attention weights.

## Results

| Method                | Mean Val. Accuracy | Best Val. Accuracy |
|-----------------------|--------------------|--------------------|
| `linear_baseline`     | 0.7354 ± 0.0151    | 0.7632             |
| `attention`           | 0.7727 ± 0.0221    | 0.8199             |
| `attention_multi_head`| 0.7622 ± 0.0223    | 0.8008             |

Results can be seen in the table above. The "Mean Val. Accuracy" column corresponds to the average validation accuracy on a 5-fold cross-validation training with early stopping, with "±" corresponding to the standard deviation across the folds. The "Best Val. Accuracy" column corresponds to the validation accuracy of the best fold. Both attention-based methods outperform the baseline, with the single-head attention model achieving the highest mean validation accuracy. The multi-head attention model, while still outperforming the baseline, does not perform as well as the single-head version. This could be due to increased model complexity leading to overfitting, given the limited amount of training data (only 4k samples). Further hyperparameter tuning and regularization techniques could potentially improve the performance of the multi-head attention model.

## Reproduction of results
To rerun the training for the `attention` method, use the following command:
```bash
python -m models.submission
```

For the `attention_multi_head` method, first update the `configs/submission.yaml` as follows:
```yaml
model:
  class_path: models.submission.Submission
  args:
    embed_dim: 3072
    latent_dim: 1024
    num_classes: 7
    multi_head: true

  best_weight_path: ckpts/best_attention_model_multi_head.pt

dataset:
  collate_fn: models.submission.attention_collate_fn
```
Then run:
```bash
python -m models.submission
```

## Submission
The method used for the submission is the `attention` model, as it achieved the best performance on the test set. The model trained on the fold with highest validation accuracy was saved and used as the submission checkpoint.