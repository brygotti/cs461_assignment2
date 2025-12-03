# Report for Graded Assignment 2 - Part 1
> CS-461 Foundation Models and Generative AI

## Methods
The provided baseline method (`norm`) implements Test-Time Adaptation by normalizing the features of each input sample using batch normalization statistics computed from the test batch itself, instead of using the running statistics accumulated during training. This approach helps to mitigate distribution shifts between training and test data by adjusting the feature distributions at test time.

To improve upon this baseline, we propose `tent`, a method based on Test-Time Entropy Minimization (Tent) as described in the paper "Tent: Fully Test-Time Adaptation by Entropy Minimization" (Wang et al., 2021). The key idea of Tent comes from the observation that performance degradation at test time is often accompanied by an increase in prediction uncertainty, or in other words, a higher entropy of the model's output distribution. Thus, Tent crafts its own objective at test time: minimizing the entropy of the model's predictions on the test data. This loss can then be used as a self-supervised signal to adapt the model parameters during test time. Only the affine parameters of the normalization layers (e.g., batch normalization layers) are updated with gradients, while batch statistics are used to compute the normalization. Other model parameters are frozen, to ensure that the model does not deviate too much from its original training configuration.

In an attempt to further improve performance, a last method called `norm_tent_three_pass` is implemented, which combines ideas from both Test-Time Normalization and Tent. Unlike the standard online setting of Test-Time Normalization or Tent, this method operates in an offline setting, utilizing much larger batches of the test set (e.g. 10'000 samples at a time) to stabilize statistics. The method performs three passes over a given test batch. For each pass, the large batch is cut into minibatches, which are fed one-by-one to the model. In the first pass, minibatches are used to compute running statistics (mean and variance) for each normalization layer, as is usually done during training. In the second pass, Tent adaptation is performed on each minibatch. It must be noted that during this second pass, the running statistics computed in the first pass are not used, but rather, minibatch statistics are used. We then perform a third pass to get the outputs, now combining the estimated running statistics with the affine parameters from Tent. We hypothesize that while Tent adapts parameters to local minibatch noise, the final inference benefits from the global stability of the accumulated running statistics over the entire batch. The offline setting is necessary here to ensure that the computed running statistics are representative of the test distribution and stable enough to be used in combination with Tent.

## Results
    
| Scenario | `unadapted` | `norm` | `tent` | `norm_tent_three_pass` |
| :--- | :--- | :--- | :--- | :--- |
| `contrast` | 0.6684 | 0.8565 | 0.8753 | 0.8824 |
| `fog` | 0.8408 | 0.8723 | 0.8891 | 0.8957 |
| `frost` | 0.8022 | 0.8212 | 0.8422 | 0.8520 |
| `gaussian_blur` | 0.7354 | 0.8810 | 0.8951 | 0.8999 |
| `pixelate` | 0.8324 | 0.8609 | 0.8808 | 0.8830 |
| `shot_noise` | 0.5682 | 0.7687 | 0.8027 | 0.8181 |
| **Mean Accuracy** | **0.7412** | **0.8434** | **0.8642** | **0.8719** |

The results show that both `norm` and `tent` significantly improve performance over the unadapted baseline across all corruption scenarios. Among the two, `tent` achieves the highest mean accuracy, demonstrating the effectiveness of entropy minimization for test-time adaptation. The combined method `norm_tent_three_pass` further improves performance, achieving the best results in all scenarios. This suggests that normalization and Tent adaptation do not always interfere with one another, and their combination can be beneficial if handled properly, particularly while using large test batches to ensure stable normalization statistics.

## Reproduction of results
For both implemented methods, hyperparameters such as learning rate, batch size, and number of adaptation steps were tuned using the provided `config_search.py` script.

## Submission
The method used for the submission is the `norm_tent_three_pass` model, as it achieved the best performance on the test set.