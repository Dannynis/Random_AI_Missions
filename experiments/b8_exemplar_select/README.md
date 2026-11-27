# B8: which examples to write into a memory classifier, on foundation features

## Origin
Random-Ideas PR #62 (`reports/2026-11-27-memory-exemplar-select`). A memory-augmented classifier (frozen 93k-param
CNN encoder, global top-3 cosine readout) grown class-incrementally to 47 EMNIST classes: storing **k-center**
(k-means medoid) exemplars instead of random ones raised final accuracy from 0.724 to 0.785 at 20/class and from 0.627
to 0.754 at 5/class (5 seeds); herding helped less; a readout-optimised swap selector overfit and doubled forgetting.

## Question
Does the selection gain survive a strong frozen encoder on natural images, and does it matter at all once the obvious
foundation-feature baseline is present — a **class-mean prototype (NCM)** that stores one vector per class and needs no
selection? (Pretrained-feature class-incremental methods are largely prototype-based.)

## Protocol
Frozen DINOv2 features (ViT-S/14, B/14, L/14; CLS + mean patch token, L2-normalised; the A2/B7 cache).
Class-incremental CIFAR-100 (10 stages × 10 classes) and CIFAR-10 (5 × 2); 10 seeds, each with its own class order and
selection randomness. When a class arrives, the selector sees its 500 / 5000 training features and writes B of them;
memory is never rewritten. Readout as in PR #62: top-3 cosine neighbours, scores Σ exp(cos/0.1). Budgets 1, 2, 5, 10,
20, 40 per class. Arms: `random`, `herding` (iCaRL), `kcenter` (spherical k-means, k-means++ init, 15 iterations,
nearest real point to each centroid), `kmeans_proto` (the centroids themselves, i.e. multi-prototype, same readout),
`all` (whole training set; upper bound), `ncm` (one class mean, nearest mean). Metrics: final accuracy over all
classes after the last stage, and average incremental accuracy.

## Pass / fail (stated before the run)
- **H1 (selection transfers):** on CIFAR-100 with ViT-B/14 features, `kcenter` beats `random` at 20/class by
  **≥0.02** final accuracy on **≥8/10** paired seeds. **Fail:** smaller gain → the PR #62 effect is a weak-encoder effect.
- **H2 (selection matters vs prototypes):** on the same cell, the best real-exemplar arm at 20/class beats `ncm` by
  **≥0.01** on **≥8/10** seeds. **Fail:** NCM ties or wins → with foundation features, which exemplars to write is
  moot and a single mean per class is the better memory.
Reported per backbone (S/B/L) to show how both effects change with encoder strength.

## Prior work
Herding vs random in class-incremental learning is mixed: herding helps some methods, others find random competitive
(Belouadah et al., arXiv 2011.01844; Masana et al. survey, arXiv 2010.15277; TEAL, arXiv 2407.00673, for small
buffers). Those studies train the backbone from scratch with replay. A frozen-foundation-feature, nearest-neighbour
memory with k-center writes against NCM prototypes is the gap this targets. It is cheap (feature extraction + kNN), so
it runs on the small tier.
