# Report for Graded Assignment 2 - Part 2
> CS-461 Foundation Models and Generative AI

## Methods
The provided baseline (`linear_baseline`) first computes an unweighted average of the embeddings of each patch before passing them through a linear layer to obtain class logits. To improve upon this, we implement two attention-based methods inspired by the paper "Attention-based Deep Multiple Instance Learning" (Ilse et al., 2018). The main idea is to allow the model to attend differently to each patch of a given sample. In other words, instead of doing a simple average of the patch embeddings, we compute a weighted average where the weights are learned through an attention mechanism.

The first method (`attention`) is a direct implementation of the attention mechanism described in the paper. It uses a two-layer neural network with tanh activation function to compute attention scores for each patch embedding. These scores are then normalized using a softmax function to obtain attention weights, which are used to compute a weighted average of the patch embeddings. This weighted average is then passed through a linear layer to obtain class logits.

The second method (`attention_multi_head`) extends the first by incorporating multi-head attention. Instead of computing a single set of attention weights, we compute one set of attention weights for each class. This should in theory increase the model's expressivity by allowing each class logit to attend to different patches in the input sample. We modify the two-layer neural network so that its output is of size `num_classes` instead of 1, and we apply the softmax function across the patches for each class separately. We then compute a weighted average of the patch embeddings for each class using the corresponding attention weights, resulting in class-specific representations. Each representation is then passed through its own linear layer to obtain the final class logits. Using different linear layers for each class is important here, so that our model stays as expressive as the single-head attention model, where each class logit also has its own set of linear weights.

Both models are trained using the same 5-fold cross-validation setup as the baseline, with early stopping based on validation loss to prevent overfitting. Hyperparameters such as learning rate and batch size are kept the same as the baseline. To allow batch training, we implement a custom collate function that pads the number of patches in each sample to the maximum number of patches in the batch. A mask is also created to indicate which patches are valid and which are padding, ensuring that the attention mechanism only considers valid patches when computing attention weights.

## Results

| Method                | Accuracy | F1 Score | Balanced Accuracy  | ROC AUC |
|-----------------------|----------|----------|--------------------|---------|
| `linear_baseline`     | 0.8083   | 0.7324   | 0.7762             | 0.9350  |
| `attention`           | 0.8773   | 0.8679   | 0.8637             | 0.9867  |
| `attention_multi_head`| 0.8428   | 0.8318   | 0.8368             | 0.9777  |

Both attention-based methods outperform the baseline linear model across all metrics. The single-head attention model achieves the highest score across metrics, indicating that allowing the model to attend differently to each patch significantly improves performance. The multi-head attention model also shows substantial improvements over the baseline, although it does not perform as well as the single-head attention model in this case. This could be due to the increased complexity of the multi-head attention model, which may require more data or tuning to fully realize its potential. It must also be noted that the multi-head attention model was early stopped a few epochs before the single-head attention model by the early stopping mechanism. These few epochs of difference in training time could have an impact on the final performance as well. Overall, these results demonstrate the effectiveness of attention mechanisms in enhancing model performance for this task.

## Submission
The method used for the submission is the `attention` model, as it achieved the best performance on the test set.