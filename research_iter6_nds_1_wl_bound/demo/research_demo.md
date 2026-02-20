# NDS 1-WL Bound

## Summary

Rigorous theoretical characterization of why NDS's nonlinear spectral coupling (alternating graph diffusion with fixed ReLU) cannot break 1-WL. Identifies three complementary algebraic obstructions: (1) fixed-point on vertex-transitive graphs where ReLU is inert, (2) permutation equivariance forcing matching feature multisets, (3) mean aggregation losing multiset information. Establishes NDS is strictly BELOW 1-WL. Novelty assessment: the specific connection between scattering-type architectures and 1-WL bounds has NOT been explicitly stated in the literature — the scattering and WL communities have operated independently. Five escape routes identified.

## Research Findings

## Why NDS Cannot Break the 1-WL Barrier: A Rigorous Characterization

### 1. The 1-WL Upper Bound for Message-Passing Architectures

The foundational result establishing the expressiveness ceiling for graph neural networks was proven independently by Xu et al. (2019) and Morris et al. (2019). Xu et al. proved that any aggregation-based GNN is at most as powerful as the 1-WL test in distinguishing different graphs (Theorem 2 in [1]), and that achieving this bound requires the aggregation function to be *injective* over multisets (Theorem 3 in [1]). Specifically, GIN achieves 1-WL power by using sum aggregation with MLPs, because sum aggregation can represent injective multiset functions (Lemma 5 in [1]) [1, 2].

Crucially, Xu et al. showed that GNNs using mean or max pooling — such as GCN and GraphSAGE — are strictly *less* powerful than the WL test. Mean aggregation maps multiple distinct multisets to the same representation: it cannot distinguish multisets that differ in cardinality but share the same mean [1, 3]. This establishes that **mean-based architectures sit strictly below 1-WL**, not merely at the 1-WL ceiling.

### 2. NDS Sits Strictly Below 1-WL

NDS (Nonlinear Diffusion Signatures) operates by alternating graph diffusion (multiplication by the normalized adjacency matrix Ā) with element-wise ReLU, starting from the degree vector. This pipeline has three properties that collectively guarantee it is *strictly less expressive* than 1-WL:

**Property 1: Fixed (non-learned) parameters.** Unlike GNNs with trainable weights, NDS uses fixed operators — the normalized adjacency Ā and a fixed ReLU. There are no learnable parameters to adapt to different graph structures. Grohe (2021) clarified that GNNs achieve 1-WL equivalence precisely because learnable parameters enable injective aggregation functions; without learning, the architecture cannot even reach 1-WL [4].

**Property 2: Mean-type aggregation.** Graph diffusion via the normalized adjacency Ā computes a degree-weighted mean of neighbor features. As established by Xu et al. [1] and further analyzed by Corso et al. (2020) [3], mean aggregation is not injective over multisets and thus cannot capture the full discriminative power of 1-WL. Sum aggregation is necessary and sufficient for 1-WL equivalence [1, 5].

**Property 3: Permutation equivariance.** The entire NDS pipeline — matrix multiplication by Ā followed by element-wise ReLU — is equivariant to permutations of the vertex set. This means that for any two graphs G₁, G₂ that are 1-WL equivalent (and thus have identical stable colorings), NDS must produce feature vectors whose multisets are identical [6, 4].

### 3. The Algebraic Obstruction: Three Complementary Arguments

#### 3.1 The Fixed-Point Argument on Vertex-Transitive Graphs

The most striking obstruction appears on vertex-transitive (VT) graphs, which form the canonical hard test cases for 1-WL. Every vertex-transitive graph is regular [7], meaning all vertices have the same degree k. The initial feature is the degree vector **d** = k·**1** (a constant vector).

The normalized adjacency matrix Ā of a k-regular graph satisfies Ā·**1** = **1** (the all-ones vector is the Perron eigenvector with eigenvalue 1) [8]. Therefore:
- Step 1: Ā·(k·**1**) = k·**1**
- ReLU(k·**1**) = k·**1** (since k > 0)
- Every subsequent step: identical output k·**1**

The degree vector is a **fixed point** of the NDS iteration. The ReLU nonlinearity never encounters negative values and is thus completely inert — it acts as the identity function. The "nonlinear" diffusion reduces to repeated linear diffusion, which on regular graphs produces a constant vector at every step [8, 7].

This is directly relevant because the classical 1-WL-hard examples are vertex-transitive strongly regular graphs. The Rook graph L(K₄) and the Shrikhande graph are both SRG(16,6,2,2), both vertex-transitive, and 1-WL cannot distinguish them [9, 10, 11]. On both graphs, NDS produces the identical constant vector 6·**1** at every iteration step. There is literally zero discriminative information.

#### 3.2 The Equivariance Argument on All Graphs

For non-vertex-transitive but 1-WL-equivalent graph pairs, the argument relies on equivariance. Arvind et al. (2019) established that 1-WL equivalence implies identical multisets of color patterns, which corresponds to an alignment of vertex orbits under automorphisms [6]. Since NDS is a deterministic, permutation-equivariant function of the adjacency matrix, it must produce the same multiset of node features on any pair of 1-WL-equivalent graphs [4, 6].

#### 3.3 The Information-Theoretic Argument (Mean < Multiset)

Even setting aside the fixed-point and equivariance issues, the use of mean aggregation (implicit in graph diffusion via Ā) fundamentally limits expressiveness. The mean function is a lossy compression of multisets: it maps {{1,1,1}} and {{1}} to the same value [1, 3]. This information loss means NDS cannot distinguish graphs that 1-WL can distinguish via its injective color refinement procedure [1, 5].

### 4. Spectral Analysis of Cross-Frequency Coupling

ReLU does mix spectral components at the node level — it is not a spectral operation and does not commute with the graph Fourier transform [13, 14]. Bastos et al. (2024) explicitly showed that standard nonlinear activation functions break functional shift equivariance in spectral GNNs [15]. However, this frequency mixing is constrained by permutation equivariance: the mixed features must still form matching multisets on 1-WL-equivalent graphs.

Under global aggregation (readout), cross-eigenmode products Σ_i φ_k(i)·φ_l(i) = δ_{kl} by orthogonality of eigenvectors [16]. Even if ReLU creates cross-frequency terms at the node level, summing over all nodes eliminates them completely.

The EPNN framework (Zhang et al. 2024) places NDS far below 3-WL in the hierarchy: MPNN < Distance-GNN ≤ EPNN < 3-WL [17]. NDS sits at the very bottom.

### 5. Novelty Assessment

The negative result about NDS is **partially but not fully known**:

**Known:** The 1-WL upper bound for MPNNs [1, 2]; mean aggregation is strictly less expressive [1]; SIGN is bounded by 1-WL [18, 19]; graph scattering transforms use similar architectures but focus on stability not expressiveness [12, 20, 21]; Pacini et al. (2025) showed all non-polynomial activations achieve same maximum separation power [22].

**NOT explicitly stated:** No paper proves interleaved diffusion + fixed ReLU cannot break 1-WL. The fixed-point argument on VT graphs has not been formulated. The unified principle "cheap + equivariant + deterministic + local + fixed ≤ 1-WL" has not been stated as a theorem, though it follows from combining [1], [4], and [22]. The scattering transform and WL-expressiveness communities have largely operated independently.

### 6. Conditions for Modified Approaches to Succeed

Five escape routes: (1) Breaking equivariance via RNI [24] or SignNet [16]; (2) Injective aggregation via sum [1]; (3) Higher-order structural features via homomorphism counts [25]; (4) Non-local operations via global attention [26]; (5) Higher-order GNNs via k-IGNs [27].

### 7. Confidence Assessment

**High confidence (>95%):** NDS is bounded by and strictly below 1-WL.
**High confidence (>90%):** The fixed-point argument on VT graphs is correct.
**Moderate confidence (70-80%):** This specific negative result has not been explicitly stated in the literature.
**What would change assessment:** Finding a paper bridging scattering transforms and 1-WL expressiveness explicitly.

## Sources

[1] [How Powerful are Graph Neural Networks? (Xu et al., ICLR 2019)](https://arxiv.org/abs/1810.00826) — Foundational paper proving GNNs are at most as powerful as 1-WL, establishing that injective (sum) aggregation achieves 1-WL while mean/max aggregation is strictly less expressive.

[2] [Weisfeiler and Leman Go Neural: Higher-order GNNs (Morris et al., AAAI 2019)](https://arxiv.org/abs/1810.02244) — Independent proof that GNNs have same expressiveness as 1-WL, and introduces k-dimensional GNNs.

[3] [Principal Neighbourhood Aggregation for Graph Nets (Corso et al., NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/99cad265a1768cc2dd013f0e740300ae-Paper.pdf) — Systematic analysis showing mean and max are strictly less expressive than sum for multiset discrimination.

[4] [The Logic of Graph Neural Networks (Grohe, 2021)](https://arxiv.org/abs/2104.14624) — Comprehensive logical characterization of GNN expressiveness via counting logics.

[5] [Some Might Say All You Need Is Sum (2023)](https://arxiv.org/abs/2302.11603) — Formal proof that sum aggregation is necessary and sufficient for 1-WL expressiveness.

[6] [On Weisfeiler-Leman Invariance (Arvind et al., 2019)](https://arxiv.org/abs/1811.04801) — Complete characterization of 1-WL and 2-WL invariant subgraph patterns.

[7] [Vertex-transitive graph (Wikipedia)](https://en.wikipedia.org/wiki/Vertex-transitive_graph) — Every vertex-transitive graph is regular, establishing constant degree vectors.

[8] [Spectra of Graphs (Brouwer & Haemers)](https://homepages.cwi.nl/~aeb/math/ipm/ipm.pdf) — For d-regular graphs, all-ones vector is Perron eigenvector with eigenvalue d.

[9] [Shrikhande graph (Wikipedia)](https://en.wikipedia.org/wiki/Shrikhande_graph) — SRG(16,6,2,2) graph cospectral with 4x4 Rook graph, canonical 1-WL-hard example.

[10] [Strongly regular graph (Wikipedia)](https://en.wikipedia.org/wiki/Strongly_regular_graph) — SRGs with same parameters are 1-WL indistinguishable.

[11] [Rook's graph (Wikipedia)](https://en.wikipedia.org/wiki/Rook_graph) — L(K4) is SRG(16,6,2,2) vertex-transitive graph paired with Shrikhande.

[12] [Stability of Graph Scattering Transforms (Gama et al., NeurIPS 2019)](https://arxiv.org/abs/1906.04784) — Proves stability bounds for graph scattering transforms, does NOT address 1-WL expressiveness.

[13] [Stability Properties of Graph Neural Networks (Gama et al., 2020)](https://arxiv.org/abs/1905.04497) — Analyzes stability of graph filters and GNNs, nonlinearities enable frequency mixing within equivariance constraints.

[14] [Graph Filters Survey](https://arxiv.org/abs/2211.08854) — Comprehensive survey of spectral graph filtering.

[15] [Equivariant ML on Graphs with Nonlinear Spectral Filters (Bastos et al., NeurIPS 2024)](https://arxiv.org/abs/2406.01249) — Shows standard nonlinear activations break graph functional shift equivariance; proposes NLSFs.

[16] [Sign and Basis Invariant Networks (Lim et al., ICML 2022)](https://arxiv.org/abs/2202.13013) — Addresses eigenvector sign ambiguity, proves universal approximation beyond standard spectral methods.

[17] [On Expressive Power of Spectral Invariant GNNs (Zhang et al., ICML 2024)](https://arxiv.org/abs/2406.04336) — Establishes EPNN framework with hierarchy: MPNN < Distance-GNN ≤ EPNN < 3-WL.

[18] [SIGN: Scalable Inception Graph Neural Networks (Frasca et al., 2020)](https://arxiv.org/abs/2004.11198) — Precomputed multi-scale diffusion without intermediate nonlinearities; bounded by 1-WL.

[19] [Scalable Expressiveness through Preprocessed Graph Perturbations (2024)](https://arxiv.org/html/2406.11714) — Confirms SIGN and simplified GNNs are bounded by 1-WL expressiveness.

[20] [Geometric Scattering for Graph Data Analysis (Gao et al., ICML 2019)](http://proceedings.mlr.press/v97/gao19e/gao19e.pdf) — Graph scattering using lazy random walk wavelets + modulus, focuses on stability not 1-WL expressiveness.

[21] [Understanding GNNs with Generalized Geometric Scattering (Perlmutter et al., SIAM 2023)](https://arxiv.org/abs/1911.06253) — Unifies graph scattering architectures with stability guarantees, does NOT characterize 1-WL expressiveness.

[22] [Separation Power of Equivariant Neural Networks (Pacini et al., ICLR 2025)](https://arxiv.org/abs/2406.08966) — All non-polynomial activations achieve same maximum separation power for given equivariant architecture.

[23] [Graph Scattering beyond Wavelet Shackles (Koke et al., NeurIPS 2022)](https://arxiv.org/abs/2301.11456) — Generalizes scattering beyond wavelets, numerical evidence of expressiveness parity.

[24] [Surprising Power of GNNs with Random Node Initialization (Abboud et al., IJCAI 2021)](https://www.ijcai.org/proceedings/2021/0291.pdf) — GNNs with random features are universal, breaking 1-WL by breaking equivariance.

[25] [Homomorphism Counts as Structural Encodings (ICLR 2025)](https://arxiv.org/abs/2410.18676) — Uses homomorphism counting to enhance GNN expressivity beyond 1-WL.

[26] [Global Self-Attention as a Replacement for Graph Convolution](https://arxiv.org/abs/2108.03348) — Global attention enables non-local information flow beyond message-passing locality.

[27] [Invariant and Equivariant Graph Networks (Maron et al., 2019)](https://arxiv.org/abs/1812.09902) — k-IGN hierarchy matching k-WL expressiveness through higher-order tensors.

[28] [Weisfeiler-Leman and Graph Spectra (Rattan & Seppelt, SODA 2023)](https://arxiv.org/abs/2103.02972) — Cospectrality is strictly finer than 2-WL; 1-WL + one individualized vertex subsumes all standard spectral invariants.

[29] [Diffusion Scattering Transforms on Graphs (Gama et al., ICLR 2019)](https://arxiv.org/abs/1806.08829) — Original diffusion scattering transform with equivariance and stability, not 1-WL bounds.

[30] [Equivariant Polynomials for Graph Neural Networks (Puny et al., ICML 2023)](https://arxiv.org/abs/2302.11556) — Equivariant polynomial basis for GNN expressiveness hierarchy.

## Follow-up Questions

- Can NDS with a non-degree initial feature vector (e.g., random node features or one-hot encodings) overcome the fixed-point obstruction on VT graphs, and if so, does the equivariance argument still prevent breaking 1-WL?
- Would replacing ReLU with the complex modulus |·| (as in classical scattering transforms) change the expressiveness characterization, given that modulus creates non-trivial frequency coupling even on regular graphs?
- Is there a formal proof that graph scattering transforms (with fixed wavelet filters + modulus + averaging) are exactly equivalent to some level of the k-WL hierarchy, or has this connection genuinely never been established?

---
*Generated by AI Inventor Pipeline*
