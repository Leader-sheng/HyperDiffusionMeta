# HyperDiffusion-Meta: Physics-Aware Metamaterial Generation via Hypergraph Neural Operators and PDE-Constrained Diffusion

![Framework Overview](asset/pipeline.png "Field-Structure Coupled Generation Pipeline")

## Introduction

To address the limitations of data-driven approaches in physical generalization, we propose **HyperDiffusion-Meta**, a novel cascade generation framework that integrates **PDE equation-constrained physical field generation** with **hypergraph-enhanced metamaterial structure generation**. This architecture aligns with the physical principles of field-structure coupling, bridging the semantic gap between text descriptions and high-precision metamaterial design.

## Core Capabilities

### Physical Field Generation

- Generate `24×24` resolution, **11-frame spatiotemporal evolution videos** of physical fields (stress, displacement, energy) from text descriptions
- PDE constraints ensure alignment with physical derivation principles
- Support training of core models and LoRA models

### Metamaterial Structure Generation

- Convert physical field dynamics into `96×96` resolution, **11-frame structural designs**
- Hypergraph-enhanced multi-body interaction modeling

- Support training of **core models**


---

**Explore our codebase to experience how HyperDiffusion-Meta bridges AI-driven design with physics-first principles!** 🚀

*(For detailed implementation and simulation results, refer to the Supplementary Materials.)*