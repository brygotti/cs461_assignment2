# Report for Graded Assignment 2 - Part 1
> CS-461 Foundation Models and Generative AI

## Methods
The provided baseline method (`norm`) implements Test-Time Adaptation by normalizing the features of each input sample using batch normalization statistics computed from the test batch itself, instead of using the running statistics accumulated during training. This approach helps to mitigate distribution shifts between training and test data by adjusting the feature distributions at test time.

To improve upon this baseline, we propose `tent`, a method based on Test-Time Entropy Minimization (Tent) as described in the paper "Tent: Fully Test-Time Adaptation by Entropy Minimization" (Wang et al., 2021). The key idea of Tent comes from the observation that performance degradation at test time is often accompanied by an increase in prediction uncertainty, or in other words, a higher entropy of the model's output distribution. Thus, Tent crafts its own objective at test time: minimizing the entropy of the model's predictions on the test data. This loss can then be used as a self-supervised signal to adapt the model parameters during test time. Only the parameters of the normalization layers (e.g., batch normalization layers) are adapted, while the rest of the model remains fixed. This ensures that the model does not deviate too much from its original training configuration.

To try and improve performance further, a last method called `norm_tent_three_pass` is implemented, which combines ideas from both Test-Time Normalization and Tent. The method only works with very large test batches (e.g. 10000 samples at a time) and performs three passes over a given test batch. For each pass, the large batch is cut into minibatches, which are fed one-by-one to the model. In the first pass, the minibatches are used to compute running statistics (mean and variance) for each normalization layer, as is usually done during training. In the second pass, each minibatch performs Tent adaptation, but while already using the computed running statistics from the previous pass instead of the running statistics from training. Finally, now that both normalization and Tent adaptation have been performed, a last forward pass is done to obtain the final outputs. The large batch size is necessary here to ensure that the computed running statistics are representative of the test distribution and stable enough to be used in combination with Tent. If the batch size is too small, normalization statistics will change for each batch, making it difficult for Tent to adapt properly.

## Results
    
| Scenario | `unadapted` | `norm` | `tent` | `norm_tent_three_pass` |
| :--- | :--- | :--- | :--- | :--- |
| `contrast` | 0.6684 | 0.8565 | 0.8673 | 0.8531 |
| `fog` | 0.8408 | 0.8723 | 0.8857 | 0.8557 |
| `frost` | 0.8022 | 0.8212 | 0.8476 | 0.8379 |
| `gaussian_blur` | 0.7354 | 0.8810 | 0.8836 | 0.8528 |
| `pixelate` | 0.8324 | 0.8609 | 0.8774 | 0.8498 |
| `shot_noise` | 0.5682 | 0.7687 | 0.8032 | 0.8078 |
| **Mean Accuracy** | **0.7412** | **0.8434** | **0.8608** | **0.8428** |

The results show that both `norm` and `tent` significantly improve performance over the unadapted baseline across all corruption scenarios. Among the two, `tent` achieves the highest mean accuracy, demonstrating the effectiveness of entropy minimization for test-time adaptation. The combined method `norm_tent_three_pass` also shows improvements over the unadapted baseline, but does not outperform `tent` alone. This suggests that while combining normalization and entropy minimization can be beneficial, the complexity introduced by the three-pass approach may not yield additional gains in this case. More extensive hyperparameter tuning or larger batch sizes might be necessary to fully leverage the potential of the combined method.

## Reproduction of results
For both implemented methods, hyperparameters such as learning rate, batch size, and number of adaptation steps were tuned using the provided `config_search.py` script.

## Submission
The method used for the submission is the `tent` model, as it achieved the best performance on the test set.