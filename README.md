# about
in this repo i'll be taking my original [nGPT](https://github.com/evintunador/nGPT) replication and speeding it up with some custom Triton kernels. I'm doing so because a potential employer found me through the nGPT replication and we decided to do a lil project in lieu of a regular coding interview

```
torchrun --nproc_per_node=2 train_ngpt.py
```