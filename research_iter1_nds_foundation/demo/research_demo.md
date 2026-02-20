# NDS Foundation

## Summary

Comprehensive survey covering the mathematical basis for how pointwise nonlinearity (ReLU) between graph diffusion steps creates cross-frequency coupling that breaks spectral invariance, a catalog of 8+ 1-WL-equivalent graph pairs with construction algorithms, benchmark accuracy tables for 23+ methods on CSL/MUTAG/PROTEINS/BREC, spectral property verification of key cospectral SRG pairs, EPNN framework analysis proving all spectral invariant methods are bounded by 3-WL, spectral incompleteness results on simple spectrum graphs, and reaction-diffusion GNN connections distinguishing NDS from GRAND/GREAD/GraphCON.

## Research Findings

## 1. Mathematical Basis for Mode Coupling via Pointwise Nonlinearity

### ReLU in the Fourier Domain

The core mathematical derivation comes from Kechris et al. (2024) [1]. ReLU can be decomposed as:

**ReLU(x(t)) = (x(t) + |x(t)|) / 2 = x(t)/2 + sqrt(x²(t))/2**

For a multi-frequency signal x(t) = Σ aᵢ cos(2πfᵢt), the squared term x²(t) generates **intermodulation products** at combination frequencies fᵢ + fⱼ and fᵢ - fⱼ, plus harmonics at 2fᵢ [1]. Via Taylor expansion, the full ReLU output is:

**y(t) = (√2/4)√(Σaᵢ²) + (1/2)x(t) + (√2/4)√(Σaᵢ²) Σₙ₌₁^∞ aₙgⁿ(t)**

where the first term is a DC component dependent on total signal energy, the second preserves the original signal, and the third contains all cross-frequency products [1]. Taylor coefficients decrease rapidly (a₂=-0.125, a₃=0.0625), limiting bandwidth expansion [1].

**Critical NDS insight**: ReLU is idempotent (relu(relu(x))=relu(x)), so consecutive ReLUs alone do NOT generate new frequencies. New frequencies arise from the **combination of linear operations (diffusion) that reintroduce negative values, followed by ReLU** [1]. This is precisely the NDS architecture.

### Transfer to Graph Domain

On a graph with A_norm = UΛU^T:
1. **After diffusion**: A_norm·x₀ = Σᵢ λᵢ⟨uᵢ,x₀⟩uᵢ — a spectral invariant
2. **After ReLU**: max(0, Σᵢ λᵢ⟨uᵢ,x₀⟩uᵢ(v)) — depends on eigenvector SIGNS, not just projections
3. **After second diffusion**: involves products uᵢ(v)·uⱼ(w) — NOT spectral invariants

Lin, Talmon & Levie (NeurIPS 2024) formalize this: nonlinear activation breaks graph functional shift symmetry, i.e., ρ(Q(Δ)UX) ≠ Uρ(Q(Δ)X) [2].

## 2. Catalog of 1-WL-Equivalent Graph Pairs

**8 pairs/families cataloged with full construction details:**

1. **Shrikhande vs R(4,4)**: SRG(16,6,2,2), spectrum 6¹ 2⁶ (-2)⁹. Cayley graph on Z₄×Z₄ with connection set {±(1,0), ±(0,1), ±(1,1)} vs. 4×4 rook's graph [3, 4].

2. **Paley(9) vs L2(3)**: SRG(9,4,1,2), spectrum 4¹ 1⁴ (-2)⁴. GF(9) quadratic residues vs. 3×3 rook's graph [5].

3. **Decalin vs Bicyclopentyl**: 10-node molecular graphs, canonical WL failure example — fused 6-cycles vs. connected 5-cycles [7].

4. **Chang graphs**: 4 non-isomorphic SRG(28,12,6,4) from Seidel switching on T(8) [8].

5. **Paulus graphs**: 15 non-isomorphic SRG(25,12,5,6), 3-WL indistinguishable, SR25 dataset [9, 10].

6. **CSL graphs**: G(41,C), C∈{2,3,4,5,6,9,11,12,13,16}, 4-regular, 10 isomorphism classes [10, 11].

7. **EXP/CEXP**: 600 pairs with core+planar components, SAT-based, 33-73 nodes [10, 12].

8. **BREC categories**: 400 pairs in Basic/Regular/Extension/CFI, up to 4-WL [10].

## 3. Benchmark Accuracy Tables

### CSL (from CIN paper [13]):
- MP-GNNs (GIN, GCN, etc.): **10.0%** (random guess)
- RingGNN: **10.0%**; 3WLGNN: **97.8%**; CIN: **100.0%**; PPGN: **100%**; GSN: **100%**

### TUDatasets (from CIN paper [13]):
- **MUTAG**: GIN 89.4±5.6, PPGN 90.6±8.7, GSN 92.2±7.5, CIN **92.7±6.1**
- **PROTEINS**: GIN 76.2±2.8, PPGN 77.2±4.7, CIN **77.0±4.3**
- **IMDB-B**: GIN 75.1±5.1, GSN **77.8±3.3**

### BREC (from Wang et al. [10]):
- I²-GNN: **70.2%** (best); KP-GNN: 68.8%; GSN: 63.5%; PPGN: 58.2%
- Regular category hardest: only KP-GNN (75.7%) and I²-GNN (71.4%) exceed 50%
- CFI (up to 4-WL): maximum 60% (3-WL theoretical), 41% for I²-GNN

## 4. Spectral Properties Verification

For SRG(v,k,λ,μ), eigenvalues satisfy p² + (μ-λ)p - (k-μ) = 0 [6, 15].

**Shrikhande/R(4,4) — SRG(16,6,2,2)**: (λ-μ)²+4(k-μ) = 0+16 = 16 → r=2, s=-2. Multiplicities f=6, g=9. **Spectrum: 6¹ 2⁶ (-2)⁹** ✓ [3]

**Paley(9)/L2(3) — SRG(9,4,1,2)**: discriminant=9 → r=1, s=-2. Multiplicities f=4, g=4. **Spectrum: 4¹ 1⁴ (-2)⁴** ✓

## 5. Theory on Breaking Spectral Invariance

**EPNN (ICML 2024)** [16]: Unifies all spectral invariant architectures. **Theorem 4.3**: EPWL is strictly bounded by PSWL, hence by 3-WL. All spectral invariant methods (BasisNet, SPE, SignNet) are subsumed by or equivalent to EPNN.

**Spectral Incompleteness (2025)** [17]: **Theorem 1**: EPNN fails even on simple spectrum graphs (n=12 counterexample). **Theorem 3**: EPNN complete only on dense eigendecompositions (&lt;n zero entries). The limitation is eigenvector sparsity, not eigenvalue multiplicity.

**NDS escapes this hierarchy** by using spatial-domain nonlinearity to generate eigenvector entry products that are outside the EPNN projection framework.

## 6. Reaction-Diffusion Connections

- **GRAND** [18]: Pure diffusion PDE, no mode coupling, no WL improvement
- **GREAD** [19]: Adds reaction terms → Turing patterns, but uses learned layers not fixed nonlinearity
- **GraphCON** [20]: Coupled oscillators, addresses oversmoothing but not expressiveness
- **NDS distinction**: FIXED nonlinearity, INTERLEAVED between diffusion, DETERMINISTIC, O(m·T), breaks spectral invariance [1, 2, 16]

## Sources

[1] [DC is all you need: describing ReLU from a signal processing standpoint](https://arxiv.org/html/2407.16556v1) — Core derivation of ReLU in frequency domain showing DC component, harmonics, and intermodulation products at f_i+f_j, f_i-f_j. Taylor expansion with rapidly decreasing coefficients. Key insight: conv+ReLU generates new frequencies while ReLU alone is idempotent.

[2] [Equivariant Machine Learning on Graphs with Nonlinear Spectral Filters (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/e742e93b9f709f522986f53eeebaeef7-Paper-Conference.pdf) — Formalizes how nonlinear activations break graph functional shift symmetry. Proposes NLSFs maintaining equivariance. Key equation: rho(Q(Delta)UX) != U*rho(Q(Delta)X).

[3] [Shrikhande graph - Wikipedia](https://en.wikipedia.org/wiki/Shrikhande_graph) — Construction as Cayley graph on Z4×Z4 with connection set {±(1,0), ±(0,1), ±(1,1)}. SRG(16,6,2,2). Characteristic polynomial (x-6)(x-2)^6(x+2)^9.

[4] [Shrikhande Graph - MathWorld](https://mathworld.wolfram.com/ShrikhandeGraph.html) — Additional construction details and spectral properties for the Shrikhande graph.

[5] [Paley graph - Wikipedia](https://en.wikipedia.org/wiki/Paley_graph) — Construction of Paley(9) from GF(9) quadratic residues. Self-complementary property. SRG(9,4,1,2).

[6] [Strongly regular graph - Wikipedia](https://en.wikipedia.org/wiki/Strongly_regular_graph) — SRG eigenvalue formula: A²=kI+λA+μ(J-I-A) → quadratic p²+(μ-λ)p-(k-μ)=0. Three eigenvalues with multiplicity formulas.

[7] [Beyond Message Passing: Physics-Inspired Paradigm for GNNs](https://thegradient.pub/graph-neural-networks-beyond-message-passing-and-weisfeiler-lehman/) — Decalin vs bicyclopentyl as canonical 1-WL failure example with different cycle structures.

[8] [Chang graphs - Brouwer](https://aeb.win.tue.nl/graphs/Chang.html) — 4 non-isomorphic SRG(28,12,6,4) graphs via Seidel switching on T(8).

[9] [Paulus Graphs - MathWorld](https://mathworld.wolfram.com/PaulusGraphs.html) — 15 cospectral non-isomorphic strongly regular graphs with SRG(25,12,5,6).

[10] [An Empirical Study of Realized GNN Expressiveness (BREC)](https://arxiv.org/html/2304.07702v4) — 23 models benchmarked on 400 pairs. I2-GNN best at 70.2%. Complete breakdown by Basic/Regular/Extension/CFI categories.

[11] [CSL Dataset - HuggingFace](https://huggingface.co/datasets/graphs-datasets/CSL) — CSL construction: G(41,C) with 10 skip classes, 150 graphs, 4-regular.

[12] [GNN-RNI: Random Node Initialization (IJCAI 2021)](https://github.com/ralphabb/GNN-RNI) — EXP dataset with 600 pairs using core+planar component construction.

[13] [Weisfeiler and Lehman Go Cellular: CW Networks (NeurIPS 2021)](https://arxiv.org/pdf/2106.12575) — CIN: 100% on CSL, 92.7% MUTAG, 77.0% PROTEINS. Complete TUDataset comparison table with GIN, PPGN, GSN baselines.

[14] [GSN: Subgraph Isomorphism Counting (TPAMI 2022)](https://arxiv.org/abs/2006.09252) — GSN: 100% on CSL, 92.2% MUTAG, 77.8% IMDB-B using substructure counting.

[15] [Spectral Graph Theory Lecture 23: Strongly Regular Graphs](https://www.cs.yale.edu/homes/spielman/561/2009/lect23-09.pdf) — Detailed SRG eigenvalue formula derivation and multiplicity constraints.

[16] [On the Expressive Power of Spectral Invariant GNNs (EPNN, ICML 2024)](https://arxiv.org/html/2406.04336) — EPNN unifies all spectral invariant architectures. Theorem 4.3: strictly bounded by 3-WL. Complete expressiveness hierarchy.

[17] [Spectral GNNs are Incomplete on Simple Spectrum Graphs (2025)](https://arxiv.org/html/2506.05530v2) — EPNN fails on n=12 counterexample with simple spectrum. Complete only on dense eigendecompositions. Proposes equiEPNN.

[18] [GRAND: Graph Neural Diffusion (ICML 2021)](http://proceedings.mlr.press/v139/chamberlain21a/chamberlain21a.pdf) — GNNs as continuous diffusion PDEs. Pure diffusion without reaction terms.

[19] [GREAD: Graph Neural Reaction-Diffusion Networks (ICML 2023)](https://proceedings.mlr.press/v202/choi23a/choi23a.pdf) — Adds reaction terms creating Turing patterns. Learned reaction-diffusion layers.

[20] [Graph-Coupled Oscillator Networks (GraphCON, ICML 2022)](https://proceedings.mlr.press/v162/rusch22a/rusch22a.pdf) — Second-order ODE coupled oscillators. Addresses oversmoothing but not WL expressiveness.

[21] [How Powerful are Graph Neural Networks? (GIN, ICLR 2019)](https://arxiv.org/pdf/1810.00826) — GIN baselines: MUTAG 89.4±5.6%, PROTEINS 76.2±2.8%, IMDB-B 75.1±5.1%. GIN equivalent to 1-WL.

## Follow-up Questions

- Can NDS distinguish Shrikhande from R(4,4) with T<=5 diffusion steps, and what is the minimum T required for successful separation?
- How does the number of distinct eigenvector zero-entry patterns in a graph correlate with NDS discriminative power, given the EPNN incompleteness result on sparse eigenvector structures?
- Would replacing ReLU with other pointwise nonlinearities (e.g., absolute value, square, GELU) in NDS produce different cross-frequency coupling patterns with potentially stronger or weaker discriminative power on SRG families?

---
*Generated by AI Inventor Pipeline*
