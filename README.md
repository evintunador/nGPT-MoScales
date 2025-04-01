# about
in this repo i'll be taking my original [nGPT](https://github.com/evintunador/nGPT) replication and speeding it up with some custom Triton kernels. I originally started this because a potential employer found me through the nGPT replication and we decided to do a lil project in lieu of a regular coding interview, however that has since fallen through.

```
torchrun --nproc_per_node=2 train_ngpt.py
```

```
torchrun --nproc_per_node=2 train_ngpt_triton.py
```

```
torchrun --nproc_per_node=2 train_MoScale.py
```