# B1: is the loss the problem, not locality? — depth composition on CIFAR

## Origin
Random-Ideas PR #46 (`reports/2026-11-11-forward-forward-negatives`) and PR #60
(`reports/2026-11-25-ff-selfsup-pairwise`). On MNIST with a 3×500 MLP those runs decomposed Forward-Forward's gap to
backprop into **locality** (free: greedy layer-local CE reached 1.49% error, *beating* the backprop MLP at 1.85%) and
**the objective** (the cost: Hinton's absolute-threshold goodness loss gave 7.26%, while a *relative* pairwise/SymBa
goodness loss closed it to 1.98%). Their sharpest negative: with a relative loss **depth added nothing** — the FF
layers "behave like a stack of independent 1-layer classifiers." Both reports' top follow-up is the test MNIST cannot
give: **a harder dataset with conv layers, where a single layer is far from sufficient and representations must
compose across depth.** That needs real GPUs and is B1.

## Hypothesis
On CIFAR-10/100, where one layer is nowhere near enough, a *layer-local* learner must compose features across depth to
do well. Two pre-stated questions:
- **H1 (parity):** does the best layer-local rule stay close to end-to-end backprop at matched conv architecture and
  epochs, as it did on MNIST — or does the gap blow up on a hard task?
- **H2 (composition):** does local training actually *use* depth (deepest-block probe ≫ first-block probe), unlike the
  flat MNIST result — or does locality cap how far features can compose?

## Setup
One conv stack, `make_blocks(depth)`: Conv3×3→(BN)→ReLU blocks, widths 64→128→256(→512), stride-2 downsample where the
width grows; global-average-pool + linear head. Three training rules at the **same architecture and epoch budget**:
- **`bp`** — end-to-end backprop (neither local nor goodness). Detached per-block linear probes ride along to give the
  depth curve for free.
- **`local_ce`** — greedy layer-local cross-entropy (LOCAL, standard loss). Each block is trained on its predecessor's
  *detached* output with its own linear CE head; earlier blocks frozen. Isolates locality.
- **`ff`** — greedy layer-local **goodness** with the pairwise SymBa loss (LOCAL + goodness loss). Supervised
  Forward-Forward: the label is written into `ncls` extra input channels, per-block goodness `g=mean(act²)`, loss
  `softplus(−4(g⁺−g⁻))` against one wrong-label negative, between-block RMS normalization, **no BN** (BN would pin the
  goodness). Readout: neutral-overlay linear probes fit on the frozen blocks. Isolates locality + the goodness loss.

Depth sweep `{4, 8, 12}` blocks, CIFAR-10 and CIFAR-100, 3 seeds. Standard crop+flip augmentation, SGD+cosine, 40
epochs per cell. Metric: test accuracy (head, and per-block probe for the depth curve).

## Pass / fail (stated before the run)
- **H1 parity (pass):** on CIFAR-10, the best local rule is within **3 pp** test accuracy of `bp` at equal depth on
  ≥2/3 seeds. (MNIST: local matched/beat backprop.) **Fail:** the gap exceeds 3 pp and widens with depth → locality
  does not scale to hard tasks.
- **H2 composition (pass):** for the best local rule, the deepest block's probe beats the first block's by **≥5 pp** on
  CIFAR-10, ≥2/3 seeds → local training composes with depth, unlike MNIST. **Fail:** the local depth curve is flat
  (<5 pp) while `bp`'s is not → locality caps composition, and the MNIST "independent 1-layer classifiers" finding
  holds on hard data.
Either outcome is reported. `bp`'s own depth curve is the reference for how much composition the architecture allows.

## Prior work
Greedy layer-wise / local learning on CIFAR is studied (Belilovsky et al., "Greedy Layerwise Learning Can Scale to
ImageNet", ICML 2019; "Decoupled Greedy Learning", 2020; Nøkland & Eidnes local-error signals, 2019). Those use
auxiliary CE/ similarity heads and report that greedy local CE scales but trails backprop by a few points at depth.
Forward-Forward's *goodness* objective at CIFAR conv depth, and a head-to-head of local-CE vs goodness vs backprop with
a controlled depth curve, is the gap this fills — and it is exactly the MNIST decomposition the daily runs asked to
scale. The local-CE arm's expected few-point gap is the quantitative baseline; the FF-goodness arm and the
composition curve are the new content.

## Outputs
`$RUN_DIR/out/summary.json`: per-cell head/block accuracies, and per-(rule, dataset, depth) aggregates incl.
`head_acc_mean`, `first_block_mean`, `best_block_mean`, `depth_gain_mean`. `run.sh tiny` is a fast CPU/GPU pipeline
check (2 depths, 2 seeds, CIFAR-10, 3 epochs, 2k train images).

## Follow-up B1b (2026-11-27): is the plateau locality, or the local objective?
The full sweep (report `reports/2026-11-27-b1-ff-depth`) found H1 FAIL (greedy local CE trails backprop by 12-17 pp
on CIFAR-10) and H2 formally PASS but with a plateau: local CE composes up to ~block 7 and then adds nothing
(CIFAR-10 d12 block probes 0.763 → 0.766 over blocks 6-12), while backprop keeps gaining to block 10. This is the
"information collapse" of greedy modules described by InfoPro (Wang et al., ICLR 2021, arXiv 2101.10832): a module
trained only on its own CE discards input information later modules need. Belilovsky et al. (ICML 2019) close most of
the gap with stronger auxiliary heads. B1b separates the two explanations at our controlled arch/epochs:
- **`local_aux`** — greedy local CE with a Belilovsky-style auxiliary head (one hidden 3×3 conv + BN + ReLU, 2×2
  average pool, linear) instead of linear-on-GAP. Tests: was the cost the weak local readout?
- **`local_rec`** — greedy local CE (linear-on-GAP head, as `local_ce`) **plus** an InfoPro-style reconstruction term:
  a 1×1 linear decoder from the block's features, bilinearly upsampled, MSE to the input image, weight 1. Tests: does
  forcing each block to keep input information lift the plateau?

Depths {8, 12} × CIFAR-10/100 × 3 seeds (same seeds/epochs as the main sweep; `bp` and `local_ce` numbers are reused
from it). Run: `B1_RULES=local_aux,local_rec B1_DEPTHS=8,12 [B1_DATASETS=cifar10] run.sh`.

**Pass / fail (stated before the run):**
- **P1 (objective, not locality):** the best of `local_aux` / `local_rec` at depth 8 on CIFAR-10 is within **3 pp** of
  `bp` d8 (0.932) on ≥2/3 seeds. **Fail:** gap >3 pp → a sizeable cost of locality remains even with the known fixes.
- **P2 (plateau lifts):** for `local_rec` at depth 12 on CIFAR-10, the block-12 probe beats the block-6 probe by
  **≥2 pp** on ≥2/3 seeds (`local_ce`: +0.3 pp). **Fail:** <2 pp → reconstruction does not restore composition at depth.
Note: `local_aux`'s readout has one extra conv layer relative to `bp`'s linear head; `bp` gains only 0.1 pp from
d8 to d12, so one extra layer cannot by itself explain parity.
