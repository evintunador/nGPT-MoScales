# about
in this repo i took my original [nGPT](https://github.com/evintunador/nGPT) replication and 1) speed it up with some custom Triton kernels then 2) implemented an idea from the video that I call "Mixture of Scales". I originally started this because a potential employer found me through the nGPT replication and we decided to do a lil project in lieu of a regular coding interview. However, that has since fallen through and all of the code & ideas are my own so I figured I'd post it.

In `kernels/`, cosine normalization and the simple residual connection work & are faster than torch.compile. In the fused logits kernel I ran into what I believe is a bug with Triton & abandoned it as sunk costs since it wasn't the focus of MoS. Then there's a residual connection meant to further integrate MoS efficiently, although I believe I only got around to doing the forward pass before deciding I had spent enough time on this project & needed to move on.

To train using any of the three models (regular nGPT, nGPT plus some triton kernels for speedup, and MoS) go into each file & edit the hyperparameters to your liking, then run one of the following commands. This code should work for 2-8 GPUs although I've only tested it on two. 
```
torchrun --nproc_per_node=2 train_ngpt.py
```
```
torchrun --nproc_per_node=2 train_ngpt_triton.py
```
```
torchrun --nproc_per_node=2 train_MoScale.py
```
