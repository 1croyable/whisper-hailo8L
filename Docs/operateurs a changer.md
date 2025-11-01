| Partie du modèle            | Opération / Formule                                   | Support Hailo-8L                          | Action recommandée                                           |
| --------------------------- | ----------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| **Prétraitement audio**     | STFT (Short-Time Fourier Transform)                   | ❌Non supporté                             | Effectuer STFT **en amont sur CPU** → entrée du NPU = *mel spectrogram* |
| **Conv front-end**          | 1D Conv / DepthwiseConv                               | ✅ Oui                                     |                                                              |
|                             | GeLu                                                  | ❌ preview                                 | SiLU                                                         |
| **Positional Encoding**     | Addition simple (x + pos)                             | ✅ Oui (`Add`)                             |                                                              |
| **LayerNorm**               | $(x - \mu)/\sqrt{\sigma^2+\epsilon}\cdot\gamma+\beta$ | ❌ Non supporté                            | On CPU                                                       |
| **Attention (Q,K,V)**       | $Q = XW_Q + b_Q$, K,V idem (Linear)                   | ✅ Oui (`MatMul`, `Add`)                   |                                                              |
|                             | Score (S = QK^T / \sqrt{d_k})                         | ❌  `MatMul` OK, division OK, mais Softmax | $\text{softmax}(QK^T)V \approx \phi(Q)(\phi(K)^TV)$où $\phi(x) = \text{SiLU}(x) + 1$ |
|                             | Softmax normalization                                 | Non supporté                              |                                                              |
|                             | Weighted sum (S_{norm}V)                              | `MatMul` OK, mais dépend du Softmax       |                                                              |
| **Feed-Forward (MLP)**      | Linear → GELU → Linear                                | ❌ GELU “preview”                          | SiLU                                                         |
| **Residual Add**            | x + y                                                 | ✅ Oui (`Add`)                             |                                                              |
| **Decoder (hors NPU)**      |                                                       | ❌ Non supportéd                           | le remplacer par CTC + un petit LM pour la score en fin      |
| **Output Softmax (logits)** | Softmax                                               | ❌ Non supporté                            | CPU, c'est pas grave, on le fait just une foi en fin         |

