# Self Pruning Neural Network

This project is part of a case study where the objective was to build a neural network that can prune its own weights during training.

Instead of training a full model and pruning later, the idea here is to let the network learn which connections are not important and gradually remove them while training itself.

---

## Approach

I implemented a custom linear layer where each weight has a corresponding gate value.

During the forward pass:

* A sigmoid function is applied to the gate scores
* The weight is multiplied by this gate

So effectively:
weight_used = weight × sigmoid(gate)

If the gate becomes very small, that weight stops contributing, which behaves like pruning.

---

## Loss Function

The total loss is a combination of:

* Cross entropy loss (for classification)
* L1 penalty on gate values (to encourage sparsity)

Total loss = CE loss + λ × sparsity loss

The L1 term pushes many gate values towards zero, which reduces the number of active weights in the network.

---

## Dataset

* CIFAR-10 dataset (loaded using torchvision)

---

## Experiments

I trained the model using different values of λ to observe how sparsity and accuracy change.

| Lambda | Test Accuracy | Sparsity (%) |
| ------ | ------------- | ------------ |
| 1e-5   |     47.23%    |    1.20%     |
| 1e-4   |     44.98%    |    1.60%     |
| 1e-3   |     42.25%    |    1.72%     |

(Results depend on training and hyperparameters)

---

## Observations

* Smaller λ → less pruning, better accuracy
* Larger λ → more pruning, but accuracy drops
* There is a clear trade-off between sparsity and performance

---

## How to Run

Install dependencies:

pip install torch torchvision matplotlib

Run the script:

python self_pruning_neural_network.py

---

## File Structure

The entire implementation is contained in a single Python file for simplicity. It includes:

* Custom prunable layer
* Model definition
* Training loop
* Evaluation and sparsity calculation

---

## Notes

* Training takes some time on CPU
* I tested with fewer epochs initially to debug
* Results improve with more epochs and tuning

---

## What I Learned

* How L1 regularization promotes sparsity
* How to modify neural network layers in PyTorch
* Trade-offs between model efficiency and accuracy
* Importance of experimenting with hyperparameters

---

## Author

Kashvee Singh
