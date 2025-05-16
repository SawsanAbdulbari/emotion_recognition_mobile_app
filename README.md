# Project Documentation: Emotion Recognition Model

## Aim of the Project

This document describes the workflow, development process, and results of a project on building and optimizing an emotion recognition model. The main goal was to identify an affective deep learning model for classifying facial expressions, fine-tune it for improved accuracy on a large dataset, and explore quantization techniques to reduce model size and document its impact on performance and latency, particularly for potential deployment on edge devices like android. The project has many phases, including initial model selection, fine-tuning with a dataset, and a detailed ablation study comparing the original and quantized model versions using both standard and custom-collected unseen data.



## Phase 1, Model Selection

 Phase 1, was focused on creating a baseline by finding the most suitable deep learning model architecture for the task of emotion recognition. This phase was done on the CSC Puhti supercomputer to use its high computational power and parallel processing possibilities, which were important for efficiently training and evaluating multiple models at the same time. The workflow was run using Slurm scripts.

### Workflow (`train.emotion_phase1.slurm`)

The `train.emotion_phase1.slurm` script managed the execution of Phase 1. Its main role was to automate the training and evaluation of a predefined set of five different model architectures: EfficientNet-B0, ResNet50, MobileNetV3-Small, MobileViT-XXS, and DenseNet121. The script began by setting up the necessary environment on the Puhti cluster, which included defining project-specific paths, locating the FER-2013 dataset (which was the dataset for this phase), and configuring directories for storing logs, trained models, performance metrics, and checkpoints. An important part of this Slurm script was its use of job arrays. This allowed for the parallel training of each model, significantly speeding up the model selection process. Each model training instance was treated as an independent task within the array, with dedicated output and error logging. The `srun` command was then used to run the `train_phase1.py` script, passing it various parameters such as the specific model to train, a configuration file (`phase1.yaml`), an instruction to utilize pretrained weights, the data directory path, batch size, and locations for output files. This  approach ensured a fair and efficient comparison of the models on the FER-2013 dataset to then select the best model for phase 2.

### Core Training Logic (`train_phase1.py`)

The Python script `train_phase1.py` was created to train deep learning architectures EfficientNet-B0, MobileNetV3-Small, ResNet50, DenseNet121, and MobileViT-XXS on the FER-2013 dataset and evaluate their performance to find the best one for emotion classification. For each of these models, the script used pretrained weights to benefit from learned features, and then edited their final classification layers to suit the specific requirements of the emotion recognition task, which in Phase 1 involved classifying images into six emotion categories: anger, fear, happy, sad, surprise, and neutral. The script was designed for flexibility, parsing command-line arguments and loading detailed configurations from an YAML file (`phase1.yaml`). Like this it was easy to switch up parameters and test different settings.

To handle data loading and preprocessing, a custom PyTorch class `EmotionDataset` was created for data augmentation for the training set. Random resized cropping, random horizontal flipping, random rotations, color jitter, and random erasing were implemented to better the diversity of the training data and so improve the model's ability to recognize unseen images. For the validation and test datasets, more standard preprocessing steps were applied, including resizing, center cropping, and normalization.

Lastly overall accuracy, as well as weighted precision, recall, and F1-score were collected, which provide a  view of the model's performance, especially understanding class imbalances. The script generated a detailed confusion matrix and, importantly, provided per-class metrics (precision, recall, F1-score for each of the six emotions).

docs/normalized_confusion_matrix.png
docs/per_class_metrics.png

The final trained model, with its configuration parameters and training history, was saved to a file, providing a complete record of each training run. `train_phase1.py` creates automated framework necessary for the training and evaluation of multiple models, giving evidence required to select the best-performing architecture for Phase 2 of the project.

---

## Phase 2, Model Finetuning and Optimization

Following the selection of EfficientNet-B0 as the best-performing model architecture from Phase 1, the project went into Phase 2. This phase was created to further refine the chosen model by fine-tuning it on a different and more comprehensive dataset, the Real-world Affective Faces Database (RAF-DB), specifically its single-label subset. Due to more balanced datadistribution we chose to use emotion category 'disgust,' expanding the classification task to seven emotions. Similar to Phase 1, Phase 2 operations were conducted on the CSC Puhti supercomputer, using its GPU resources for afficient training. The workflow was again managed by a Slurm script, and the fine-tuned model was prepared for quantization and detailed performance analysis.

### Workflow (`train.emotion_phase2.slurm`)

The `train.emotion_phase2.slurm` script was responsible for managing the execution of the fine-tuning process in Phase 2. Its main objective was to take the EfficientNet-B0 model, found optimal in Phase 1, and further train it using the RAF-DB dataset with a specific set of hyperparameters to enhance its emotion recognition capabilities. The script configured the Puhti environment, requesting necessary computational resources such as a V100 GPU, an appropriate number of CPUs, and sufficient memory. 

An important aspect of this Slurm script was the definition of hyperparameters tailored for this phase. The specified hyperparameters for the final successful run were: `--model efficientnet_b0 --batch_size 16 --epochs 30 --lr 0.0005 --weight_decay 1e-3`. The script also handled other training-related parameters such as the number of data loader workers, mixup alpha for data augmentation, early stopping patience, and a label smoothing factor.

The core execution step involved runing the Phase 2 Python training script (`train_phase2local.py`) using `srun`. All the predefined parameters, paths, and hyperparameters were passed as command-line arguments to this Python script. 

Upon completion of the training script, the Slurm script was designed to allow following scripts, particularly the model optimization and quantization script, to easily and automatically locate the most recent and relevant fine-tuned model. 


### Core Training Logic (`train_phase2local.py`)

The Python script `train_phase2local.py`'s main purpose was to take the EfficientNet-B0 model, which had been identified as the top performer in Phase 1, and further enhance its capabilities by fine-tuning it on the RAF-DB dataset. The script was engineered to be more developed than its Phase 1 counterpart and incorporated a range of advanced training techniques to maximize performance.

`EmotionDataset` class was again used but adapted for RAF-DB, and it was specifically designed to handle its unique label mapping (where numeric labels 1-7 correspond to specific emotion strings).

The script defined seven emotion classes for this phase: 'anger', 'fear', 'happy', 'sad', 'surprise', 'neutral', and 'disgust'. While the main model for fine-tuning was EfficientNet-B0, the script also contained code for an `EmotionAttentionNetwork`. This custom architecture, which could leverage EfficientNet-B0 or EfficientNet-B2 as a backbone, was designed to incorporate attention mechanisms such as CBAM (Convolutional Block Attention Module) and SEBlock (Squeeze-and-Excitation Block). These attention mechanisms were intended to help the model focus on potentially improving feature extraction and emotion recognition accuracy. 

The augmentation pipeline for the training data included techniques such as resizing, random resized cropping, random horizontal flipping, random affine transformations, and color jitter. Normalization was applied consistently. For the validation set, a less aggressive set of transformations, like  resizing, center cropping, and normalization, was used to ensure a fair evaluation of the model's learned capabilities.

The training included mixup data augmentation, a technique that trains the model on linear interpolations of pairs of examples and their labels, potentially leading to improved generalization. Label smoothing was applied to the CrossEntropyLoss function, a regularization technique that helps prevent the model from becoming overconfident. To ensure training stability, gradient clipping was used to prevent the gradients from becoming too large. An early stopping was also used and it would stop the training process if no improvement was sen for 10  epochs, saving computational resources and preventing overfitting. The best model weights found during training were saved. A learning rate scheduler was employed to dynamically adjust the learning rate, and the AdamW optimizer was used.

The fine-tuned model's performance on the test or validation set was calculated and reported a comprehensive set of metrics, including overall accuracy, weighted precision, recall, and F1-score, along with a detailed confusion matrix.

It also supported TensorBoard logging for real-time visualization of training metrics. After completion, the final fine-tuned model, along with the optimizer state, the history of training metrics, and relevant configuration details, was saved to a timestamped output directory for later analysis and for use in the quantization phase.

---

### Model Optimization and Ablation Study (`optimize_model_original.py`)

The `optimize_model_original.py` scripts focus was on optimizing the fine-tuned emotion recognition model from Phase 2 and conducting a detailed performance comparison. The main objective was to take the highly accurate EfficientNet-B0 model, apply quantization to reduce its size and then do an ablation study. This study was designed to meticulously compare the performance accuracy and inference latency—of the original full-precision (FP32) model against its quantized (INT8) counterpart.

It began by loading the trained PyTorch model checkpoint that was saved at Phase 2. During the loading process, it dynamically found the model architecture (e.g., `efficientnet_b0`) and the number of emotion classes directly from the information stored in the checkpoint file.

The script then implemented dynamic INT8 quantization for the loaded model. The quantization process specifically targeted the linear and convolutional layers of the neural network and was performed on the CPU.

To measure performance metrics, inference latency, it calculated the file size, in megabytes (MB), of both the original FP32 model and the quantized INT8 model. This involved running inference 30 times on a given input tensor and recording the average, minimum, and maximum inference times in milliseconds. 

An evaluation function was also defined to understand the predictive performance of both the original and quantized models on a given dataset. This function calculated standard classification metrics, including accuracy, and weighted precision, recall, and F1-score. These metrics were important for understanding any impact of quantization on the model's ability to correctly classify emotions and were used in the ablation study.

For the ablation study itself, the script was designed to handle multiple datasets. It loaded and utilized both the RAF-DB dataset and a custom dataset. This custom dataset, with images manually collected from Pexels, was specifically included to serve as unseen data. The script created PyTorch DataLoaders for both these datasets to feed images to the models during the study. In each experiment, a batch of images was processed, with the script alternating between taking images from the RAF-DB dataset and the custom Pexels dataset. For every image within each batch, the script recorded the predictions made by both the original and the quantized models, whether these predictions were correct when compared against the true labels, and the inference latency for both models on that specific batch.

---

## Results

Following the successful completion of Phase 2, which involved fine-tuning the EfficientNet-B0 model with the RAF-DB dataset and optimized hyperparameters (`--model efficientnet_b0 --batch_size 16 --epochs 30 --lr 0.0005 --weight_decay 1e-3`), the project achieved great performance in emotion recognition in seven emotion categories. The evaluation of this fine-tuned model gave detailed metrics for each emotion, reflecting an understanding of facial expressions.

For the emotion 'anger,' the model demonstrated a precision of 0.7751, a recall of 0.7360, and an F1-score of 0.7550. This indicates a strong ability to correctly identify instances of anger while maintaining a good balance between precision and recall.

In recognizing 'fear,' the model achieved a high precision of 0.7872. However, the recall was 0.5068, leading to an F1-score of 0.6167. While the model was accurate when it predicted fear, it missed a notable portion of actual fear instances.

The model showed great performance for the 'happy' emotion, with a precision of 0.9279, recall of 0.9381, and an F1-score of 0.9329. These high scores tell that the model is very effective and reliable in identifying happiness.

For 'sad' expressions, the model had a precision of 0.7833, a recall of 0.8408, and an F1-score of 0.8110. The high recall indicates that the model was good at capturing most instances of sadness.

The 'surprise' emotion was also well-recognized, with a precision of 0.8394, recall of 0.8629, and an F1-score of 0.8510, demonstrating a strong and balanced performance for this category.

For 'neutral' expressions, the model achieved a precision of 0.8115, recall of 0.8264, and an F1-score of 0.8189, indicating consistent and reliable identification.

Lastly, for the 'disgust' emotion, the model registered a precision of 0.6879, recall of 0.5511, and an F1-score of 0.6120. Similar to 'fear,' while the precision is reasonable, the recall suggests that the model found 'disgust' more challenging to detect comprehensively, and that instances of disgust were less represented in the dataset compared to other emotions.

Overall, these results highlight the success of the fine-tuning phase in developing a capable emotion recognition model, with particularly strong performance for happiness, surprise, sadness, and neutral expressions, and identifies areas such as fear and disgust where further targeted improvements could be beneficial.


## Model Quantization and Ablation Study Findings

Following the fine-tuning of the EfficientNet-B0 model, a critical phase of the project involved model quantization and a ablation study to evaluate its impact. The main goals of quantization were to achieve a smaller model size suitable for potential mobile deployment, while trying to maintain high accuracy and not significantly compromise latency. The `optimize_model_original.py` script was responsible for performing dynamic INT8 quantization on the fine-tuned model and then doing this ablation study.

The ablation study had 30 experiments. These experiments were designed to compare the performance of the original full-precision (FP32) model with its quantized (INT8) model. The evaluation was performed using batches of images drawn alternately from two datasets, the RAF-DB dataset (on which the model was fine-tuned in Phase 2) and a custom dataset. This custom dataset, manually collected from Pexels, acted as unseen data to assess the models' generalization capabilities.

Key findings from the ablation study, which detailed the per-experiment batch accuracies and latencies, are as follows:

1.  **Accuracy Preservation**: Across the 30 experiments, the dynamic INT8 quantization had success in preserving the predictive accuracy of the model. For numerous batches, particularly from the RAF-DB dataset, the batch accuracy of the quantized model was identical to that of the original model. Similarly to the custom (unseen) dataset, where both models typically achieved identical, sometimes lower, batch accuracies. This indicates that the quantization process did not lead to a significant degradation in the model's ability to correctly classify emotions, even on previously unseen data.

2.  **Latency Considerations**: The ablation study reported latency measurements for both models. The original FP32 model was typically evaluated on a GPU (if available), while the quantized INT8 model was evaluated on the CPU. Under this specific comparison setup (GPU FP32 vs. CPU INT8), the quantized model exhibited noticeably higher average inference latencies per batch, where original model latencies were often in the range of 5-10 ms, while quantized model latencies were frequently in the 40-50 ms range. This is an important observation. While one of the goals was not to compromise latency, this comparison highlights the performance difference between GPU and CPU execution. If the target deployment is CPU-bound, the relevant comparison would be FP32 CPU vs. INT8 CPU, which might show a latency benefit for the INT8 model on CPU. The provided study focuses on the GPU vs. CPU scenario.

3.  **Model Size Reduction**: There was a big reduction in model file size. This is a significant advantage for deployment on resource constrained environments like mobile devices.

In summary, the ablation study, did 30 experiments with both known (RAF-DB) and unseen (custom Pexels) data, indicated that dynamic INT8 quantization was highly effective in maintaining the accuracy of the fine-tuned EfficientNet-B0 model. The main trade-off obacted in the provided logs, when comparing GPU-run original model versus CPU-run quantized model, was an increase in latency for the quantized version. The result shows the viability of the quantized model in terms of accuracy retention, which, coupled with the expected benefit of reduced size, makes it a for further deployment considerations, especially if CPU-based inference is the target or if the size reduction is paramount.




## References

1.  **FER-2013 Dataset**: Retrieved from [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

2.  **RAF-DB (Real-world Affective Faces Database)**: Li, S., Deng, W., & Du, J. (2017). Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2252-2260). DOI: 10.1109/CVPR.2017.243. The dataset is available from [http://www.whdeng.cn/RAF/model1.html](http://www.whdeng.cn/RAF/model1.html).

3.  **Custom Dataset Source (Pexels)**: Pexels. (n.d.). *Free Stock Photos, Royalty Free Stock Images & Copyright Free Pictures*. Retrieved May 16, 2025, from [https://www.pexels.com](https://www.pexels.com). (Images used for the custom dataset were manually collected from Pexels, which provides royalty-free stock photos under the Pexels license).