# UniGenCoder: Merging SEQ2SEQ and SEQ2TREE Paradigms for Unified Code Generation

This is the official PyTorch implementation for the following ICSE2025 NIER paper:

**Title**: [UniGenCoder: Merging SEQ2SEQ and SEQ2TREE Paradigms for Unified Code Generation](https://arxiv.org/pdf/2502.12490)

Our implementation is built on the source code from [CodeT5](https://github.com/salesforce/CodeT5) and [Tranx](https://github.com/pcyin/tranX). Thanks for their work.

## Environment
We recommend readers to refer to `envs.yaml`(conda export) or `envs.txt`(pip export) for more detailed environment information.

## Dataset

| \--task   | \--sub\_task                       | Description                                                                                                                      |
| --------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| code generation   | nl-java                               | text-to-code generation on [Concode](https://aclanthology.org/D18-1192.pdf) data                                                 |
| code translation | cs-java                    | code-to-code translation from [C# to Java](https://arxiv.org/pdf/2102.04664.pdf)                                             |

## Download


* [data and checkpoints (under preparing)](???)


## Quick Start

1. Before starting, please place data in the right position. The correct `data` directory structure should be:

   ```bash
   data
   ├── concode
       ├── train.json
       ├── dev.json
       └── test.json
   ```

2. Generate the grammar file.

   ```bash 
   cd UniGenCoder
   python build_ast.py
   ```

3. Prepare the seq2seq teacher and seq2tree teacher and get corresponding best model using checkpoint average: 

   ```bash
   cd CodeT5
   bash sh/fine_tune_concode.sh
   bash sh/average_model.sh
   bash sh/fine_tune_concode_tree.sh
   bash sh/average_model.sh
   ```

4. UniGenCoder backbone training:

   ```bash
   bash multitask_sh/multitask_distill_cross.sh
   bash sh/average_model.sh
   ```

5. Prepare data for the selector:

   ```bash
   bash sh/multitask_distill_cross_inference.sh
   <!-- modify test_split_tag to prepare for different dataset -->
   ```


6. UniGenCoder selector training and inference:

   ```bash
   bash multitask_sh/multitask_distill_cross_tune.sh
   ```


## Citation

If you find this code to be useful for your research, please consider citing:

```
@article{DBLP:journals/corr/abs-2502-12490,
  author       = {Liangying Shao and
                  Yanfu Yan and
                  Denys Poshyvanyk and
                  Jinsong Su},
  title        = {UniGenCoder: Merging Seq2Seq and Seq2Tree Paradigms for Unified Code
                  Generation},
  journal      = {CoRR},
  volume       = {abs/2502.12490},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2502.12490},
  doi          = {10.48550/ARXIV.2502.12490},
  eprinttype    = {arXiv},
  eprint       = {2502.12490},
  timestamp    = {Wed, 19 Mar 2025 11:49:46 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2502-12490.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```