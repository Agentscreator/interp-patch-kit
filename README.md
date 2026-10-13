# interp-patch-kit

A lightweight tool for running activation patching sweeps on transformer language models, built on
TransformerLens.

## Why

Activation patching means testing each layer and attention head one at a time, which is slow and
repetitive to do by hand. This tool runs the whole sweep for you, so you can focus on interpreting
the results instead of writing patching loops.

![Patching heatmap for the indirect-object task on GPT-2 Small](example_heatmap.png)

## Install

```bash
pip install git+https://github.com/Agentscreator/interp-patch-kit.git
```
