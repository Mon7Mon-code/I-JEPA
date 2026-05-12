# I-JEPA
JEPA is a self-supervised ML architecture, introduced in 2022, that learns semantic representations from data. It's a non-generative model that predicts missing information in embedding space rather than reconstructing its inputs, and is currently one of the leading candidates to be used as world models for robotic AI and autonomous machine intelligence, due to its more efficient design and ability to learn relevant abstract representations.

I-JEPA is Meta AI's 2023 implementation for JEPA for images, which predicts the representations of masked regions within an image, using Vision Transformers within its architecture.

## Architecture
I-JEPA consists of 3 components:
  - **Context Encoder**: Processes a masked view of an image into representations within embedding space
  - **Target Encoder**: Processes the full unmasked image into embedding space. Its weights are an EMA of the context encoder to prevent representation collapse
  - **Predictor**: Takes the input from the context encoder and positional mask tokens to predict the target encoder's embeddings

The context and target encoders are ViT models with 12 transformer blocks with each having 6 attention heads running in parallel. The predictor is a ViT model that consists of 6 transformer blocks with all three models having an embedding dimension of 384.

The masking strategy used follows what's in the paper; 4 target blocks at scale 0.15-0.2 and 1 context block at scale 0.8-1.0. The loss function is the MSE of the normalised predicted and target representations in embedding space.

## Implementation
Due to hardware constraints (1 RTX 3060 Ti 8GB), I made changes to the original implementation to ensure a model can be trained within reasonable time. These include:
  - **Dataset**: STL-10 unlabelled (100k images) (vs ImageNet 1.2M)
  - **Epochs**: 100 (vs 300)
  - **Batch size**: 32 (vs 2048)
  - **Model**: ViT-Small (vs ViT-H)

Despite these initial changes, a single epoch still took over half an hour to complete. Hence, I implemented FP16 mixed precision training instead of BF16 in order to speed up training, while having a negligible decrease in the model's performance.

A bug to note that affected the implementation is that I didn't code the EMA momentumn value annealing to 1.0; instead it's fixed at 0.996.

## Results
As in the paper, I evaluated the model using a linear probe to predict the labels of images within the STL-10 dataset. The training setup for the probe consists of:
  - **Loss Function**: Cross Entropy Loss
  - **Optimiser**: SGD (Momentum: 0.9, Weight Decay: 0.0, Batch size: 64, Epochs: 50)
  - **LR Scheduler**: LR decays by 0.1 every 15 epochs

From testing, the best performance I obtained before plateauing was with an initial learning rate of 0.4. The accuracy of the linear probe was **61.59%** with this learning rate. Compared to the paper, it's lower but it still suggests that the model has learned significant semantic representations of the images despite a significantly smaller dataset than ImageNet 1.2M. This is as the accuracy is around 6 times greater than the random baseline of 10% for choosing the correct class.

Something to take note of, normalisation of the representation vector consistently hurt the accuracy of the probe. It suggests that in this model, the magnitude of the embeddings matter as much as its directions in embedding space in representing images. This is expected to be seen in models which haven't been able to encode representations fully using directions in embedding space due to undertraining.

## Usage
I used Visual Studio Code to create and run this model.

1. Run `pip install torch torchvision` in the terminal
2. Run train.py (This will run for 100 epochs and have checkpoints every 10 epochs. Progress is printed in terminal)
3. Run eval.py (Ensure that the checkpoint path in the code points to the right folder)

## References
- Assran et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture: https://arxiv.org/abs/2301.08243
- Dosovitskiy et al. (2020). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale: https://arxiv.org/abs/2010.11929
- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf




















