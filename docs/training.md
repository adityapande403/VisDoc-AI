# Contrastive Fine-Tuning & Hard Negative Mining

## 1. Contrastive Training Objective

VisDoc AI fine-tunes visual document retrieval models using a ColBERT-adapted multi-vector contrastive objective:
$$\mathcal{L}_{\text{total}} = (1 - w) \mathcal{L}_{\text{pairwise}} + w \mathcal{L}_{\text{in-batch}}$$

### 1.1 Pairwise Negative Cross-Entropy Loss
For each query $Q$, positive document page $D^+$, and mined hard negative page $D^-$:
$$s^+ = \operatorname{MaxSim}(Q, D^+), \quad s^- = \operatorname{MaxSim}(Q, D^-)$$
$$\mathcal{L}_{\text{pairwise}} = \log\left(1 + \exp\left(-\frac{s^+ - s^-}{\tau}\right)\right)$$
where $\tau = 0.02$ is the temperature scaling hyperparameter.

### 1.2 In-Batch Negatives
For batch size $B$, non-corresponding document pages in the same mini-batch act as negative pairs:
$$S_{ij} = \operatorname{MaxSim}(Q_i, D_j^+)$$
$$\mathcal{L}_{\text{in-batch}} = \operatorname{CrossEntropyLoss}\left(\frac{S}{\tau}, \text{labels}=[0, 1, \dots, B-1]\right)$$

---

## 2. Parameter-Efficient Fine-Tuning (PEFT / LoRA)

To balance computational efficiency and retrieval gains:
- **Projection Head Adaptation**: Trainable linear layer `custom_text_proj: Linear(2048 -> 128)`.
- **LoRA Modules**: Low-Rank Adaptation applied to attention projection layers (`q_proj`, `v_proj`, `down_proj`, `gate_proj`).
- **Precision**: `bfloat16` mixed precision for optimal numerical stability on Blackwell / RTX 50-series GPUs.

---

## 3. Hard Negative Mining Pipeline

Hard negatives prevent model collapse and teach the model fine-grained discrimination between adjacent pages or similar documents.
1. Encode current collection using baseline checkpoint.
2. Query the index for top-$k$ nearest neighbors.
3. Filter out known positive page IDs.
4. Categorize negative difficulty:
   - `adjacent_page`: Different page from the same document.
   - `hard`: Cross-document candidate with high MaxSim score ($s > 0.3$).
   - `medium`: General semantic negative.
5. Save enriched dataset with complete provenance.
