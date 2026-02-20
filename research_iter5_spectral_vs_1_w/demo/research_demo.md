# Spectral vs 1-WL

## Summary

Comprehensive analysis of why spectral coupling energy from nonlinear graph diffusion (NDS: iterative ReLU∘normalized-adjacency on degree features) cannot break the 1-WL expressiveness ceiling. Three reinforcing failure mechanisms identified: (1) fixed-point trap on regular/vertex-transitive graphs where degree vector is invariant under the iteration, producing zero coupling; (2) permutation equivariance of all NDS operations forces features to respect automorphisms, bounding expressiveness at 1-WL; (3) mean aggregation loses multiset information making NDS strictly weaker than 1-WL. Supported by 19 sources spanning GNN theory, spectral graph signal processing, and WL hierarchy literature.

## Research Findings

## Why Spectral Coupling Energy from Nonlinear Diffusion Fails to Translate into Graph Distinguishing Power

### Executive Summary

The Nonlinear Diffusion Signature (NDS) — defined as iterative application of x_{t+1} = ReLU(Ā · x_t) starting from degree vector x_0 = d, where Ā is the normalized adjacency matrix — produces mathematically real cross-frequency coupling between Laplacian eigenmodes through the pointwise ReLU nonlinearity. However, this coupling fundamentally cannot break the 1-Weisfeiler-Leman (1-WL) expressiveness ceiling. The failure has three reinforcing explanations at different levels of analysis.

---

### LEVEL 1: The Fixed-Point Trap — No Coupling Occurs on the Hardest Graphs

The most devastating failure mode is that on precisely the graphs where distinguishing power is most needed — regular and vertex-transitive graphs including all strongly regular graphs (SRGs) — the NDS iteration produces **zero coupling energy** because the degree vector is a fixed point.

**The Perron eigenvector argument.** For a connected d-regular graph, the normalized adjacency matrix Ā = D⁻¹A = (1/d)A has the all-ones vector **1** as its Perron eigenvector with eigenvalue 1 [1, 2]. The degree vector on a d-regular graph is d·**1**, proportional to this Perron eigenvector. Therefore Ā·d = d, and since all entries of d are positive, ReLU(d) = d. By induction, x_t = d for all t ≥ 0.

**Why this matters for expressiveness.** Strongly regular graphs are the canonical hard cases for graph isomorphism — any two SRGs with the same parameters (n, k, λ, μ) are indistinguishable by 1-WL and even 2-WL [3]. These are exactly vertex-transitive d-regular graphs where NDS features remain constant. The spectral coupling energy is literally zero: the coupling integral Σ_i φ_k(i)·φ_l(i)·f(i), where f is the constant node feature vector, reduces to c·δ_{kl} by eigenvector orthogonality [4, 7].

**Oversmoothing connection.** This is a special case of oversmoothing [5, 6]: repeated normalized adjacency multiplication causes node features to converge to the dominant eigenspace. On regular graphs, convergence is immediate because the degree vector already lies in the dominant eigenspace.

---

### LEVEL 2: The Equivariance Constraint — Coupling Cannot Break Automorphism Symmetries

Even on non-regular graphs where NDS features evolve across iterations, expressiveness is still bounded by 1-WL because all NDS operations are **permutation equivariant**.

**Pointwise nonlinearity preserves equivariance.** A function σ: ℝ → ℝ applied elementwise is permutation equivariant: for any permutation matrix P, σ(Px) = Pσ(x) [7, 8]. Both normalized adjacency multiplication and pointwise ReLU are permutation equivariant, and their composition preserves this property.

**Automorphic nodes must receive identical features.** Pearce-Crump and Knottenbelt (2024) established that any deterministic Aut(G)-equivariant function must assign identical outputs to automorphically equivalent nodes [9]. Since NDS is deterministic and equivariant, it cannot distinguish automorphic node pairs.

**NDS as a fixed-weight GNN.** The NDS iteration is precisely a single-layer GNN with identity weight matrix, no bias, mean aggregation, and ReLU activation [1, 10]. Xu et al. (2019) proved that message-passing GNNs are bounded by 1-WL [1]. NDS is a special case, so NDS ⊆ 1-WL.

**Frequency mixing is real but equivariant.** Gama et al. (2020) proved that pointwise nonlinearities create genuine cross-frequency coupling — energy scatters between Laplacian eigenmodes [7]. The Isufi et al. (2024) survey confirmed this frequency mixing effect [11]. However, this mixing is itself permutation-equivariant: it produces the same redistribution pattern for automorphically equivalent nodes.

**Contrast with NLSF.** Bastos et al. (NeurIPS 2024) identified that standard pointwise nonlinearities break *functional shift equivariance* (a finer symmetry than permutation equivariance) WITHOUT gaining expressiveness [12]. ReLU breaks an algebraic structure without gaining discriminative power because it still preserves the coarser permutation equivariance that constrains 1-WL expressiveness.

---

### LEVEL 3: Information-Theoretic Loss — Mean Aggregation Loses Multiset Information

NDS has a fundamental information-theoretic disadvantage compared to even 1-WL: it uses **mean aggregation** rather than **injective multiset aggregation**.

**1-WL uses injective aggregation** via hash functions on multisets [1]. **Mean aggregation is not injective** — it maps {1,1,1} and {1} to the same value, losing cardinality information [1]. The normalized adjacency in NDS computes neighborhood means, so NDS loses information that 1-WL retains.

**NDS ⊊ 1-WL (strictly weaker).** Combining the equivariance bound (NDS ≤ 1-WL) with mean aggregation's information loss, NDS is strictly weaker than 1-WL on general graphs. Wu et al. (2019) demonstrated that SGC (removing nonlinearities entirely) achieves competitive practical performance despite being a linear model [13], suggesting the nonlinearity adds surprisingly little discriminative value in the diffusion context.

---

### LEVEL 4: Spectral Invariant Theory — Algebraic Constraints

**EPNN bounds.** Zhang et al. (2024) proved all spectral invariant GNNs are strictly less expressive than 3-WL (Theorem 4.3, Corollary 4.5) [4]. NDS is vastly weaker than EPNN since it accesses no eigenspace projections or spectral distances — only iterative diffusion features.

**Sign ambiguity.** Lim et al. (2022) identified eigenvector sign/basis ambiguity as fundamental challenges for spectral methods [14]. Hordan et al. (2025) proved spectral GNNs are incomplete even on simple-spectrum graphs [15].

---

### CONDITIONS UNDER WHICH COUPLING COULD HELP

1. **Random Node Initialization**: Abboud et al. (2021) proved GNNs with RNI are universal [16] — random features break equivariance, enabling distinction of automorphic nodes.
2. **Spatio-Spectral interactions**: Geisler et al. (NeurIPS 2024) showed S²GNNs with spectral positional encodings exceed 1-WL [17].
3. **Higher-order multiplicative interactions**: Beyond-1-WL requires multiplicative interactions between DIFFERENT feature vectors, not reshaping a single vector via ReLU [4, 17].
4. **Learned weights**: Adding learnable parameters makes it a standard GNN, approaching (but not exceeding) 1-WL with appropriate architecture [1].

---

### CONTRADICTING EVIDENCE

**Practical utility despite limits.** SGC achieves competitive accuracy on many benchmarks despite being linear [13], suggesting theoretical expressiveness limits are less relevant for practical tasks.

**Frequency mixing IS real.** It would be incorrect to claim ReLU does nothing spectrally — Gama et al. (2020) rigorously proved genuine cross-frequency coupling exists [7]. The coupling is real in signal processing terms; it just cannot create new graph invariants.

**Non-regular graph dynamics.** On non-regular graphs, NDS produces evolving non-constant features with genuine nonlinear effects. However, the resulting features remain bounded by 1-WL.

---

### CONFIDENCE: High (>90%)

The three-level explanation is supported by established results from Xu et al. [1], Gama et al. [7], and Pearce-Crump [9]. The fixed-point argument follows from elementary linear algebra. What would change this: discovery that NDS distinguishes some graph pair that 1-WL cannot — but this would contradict Xu et al.'s theorem since NDS is a special case of mean-aggregation GNN.

## Sources

[1] [How Powerful are Graph Neural Networks? (Xu et al., ICLR 2019)](https://arxiv.org/abs/1810.00826) — Established that message-passing GNNs are bounded by 1-WL; proved mean/max aggregation captures strictly less information than sum aggregation; introduced GIN.

[2] [Laplacian Matrix — spectral properties of graph matrices](https://en.wikipedia.org/wiki/Laplacian_matrix) — Reference for Perron-Frobenius eigenvector properties of adjacency and normalized adjacency matrices on regular graphs.

[3] [The Weisfeiler-Leman Algorithm and Recognition of Graph Properties (2021)](https://arxiv.org/pdf/2005.08887) — Established that SRGs with same parameters are 2-WL-equivalent; vertex-transitive graphs challenge k-WL.

[4] [On the Expressive Power of Spectral Invariant Graph Neural Networks (Zhang et al., 2024)](https://arxiv.org/abs/2406.04336) — Introduced EPNN unifying framework; proved all spectral invariant GNNs strictly less expressive than 3-WL.

[5] [Oversmoothing in GNNs: why does it happen so fast?](https://xinyiwu98.github.io/posts/oversmoothing-in-shallow-gnns/) — Analysis of convergence to dominant eigenspace under repeated normalized adjacency multiplication.

[6] [GNN: Over-smoothing (ETH Zurich seminar)](https://disco.ethz.ch/courses/fs21/seminar/talks/GNN_Oversmoothing.pdf) — Theoretical characterization of oversmoothing as convergence to invariant subspace with exponential decay.

[7] [Stability Properties of Graph Neural Networks (Gama et al., IEEE TSP 2020)](https://arxiv.org/abs/1905.04497) — Proved pointwise nonlinearities create cross-frequency coupling (frequency mixing) in GNNs.

[8] [GNN: Architectures, Stability, and Transferability (Ruiz, Gama, Ribeiro, 2021)](https://arxiv.org/abs/2008.01767) — Proved GNNs exhibit permutation equivariance and stability to deformations; scattering from nonlinearity.

[9] [Graph Automorphism Group Equivariant Neural Networks (Pearce-Crump, ICML 2024)](https://proceedings.mlr.press/v235/pearce-crump24a.html) — Characterized Aut(G)-equivariant functions; automorphic nodes must receive identical outputs.

[10] [Semi-Supervised Classification with GCN (Kipf & Welling, ICLR 2017)](https://openreview.net/pdf?id=SJU4ayYgl) — Introduced GCN with normalized adjacency propagation rule; NDS is GCN with identity weights.

[11] [Graph Filters for Signal Processing and ML on Graphs (Isufi et al., IEEE TSP 2024)](https://arxiv.org/abs/2211.08854) — Survey confirming GCNNs as nonlinear graph filters with frequency mixing; spectral equivalence.

[12] [Equivariant ML on Graphs with Nonlinear Spectral Filters (Bastos et al., NeurIPS 2024)](https://arxiv.org/abs/2406.01249) — Standard pointwise nonlinearities break functional shift equivariance without adding expressiveness; proposed NLSFs.

[13] [Simplifying Graph Convolutional Networks (Wu et al., ICML 2019)](https://arxiv.org/abs/1902.07153) — Removing nonlinearities from GCN achieves competitive performance; nonlinearity adds little discriminative value.

[14] [Sign and Basis Invariant Networks (Lim et al., ICML 2022)](https://arxiv.org/abs/2202.13013) — Identified eigenvector sign/basis ambiguity; proposed SignNet/BasisNet with universal approximation.

[15] [Spectral GNNs are Incomplete on Simple Spectrum Graphs (Hordan et al., 2025)](https://arxiv.org/abs/2506.05530) — Proved spectral GNNs incomplete even on simple-spectrum graphs; proposed equiEPNN.

[16] [Surprising Power of GNNs with Random Node Initialization (Abboud et al., IJCAI 2021)](https://arxiv.org/abs/2010.01179) — GNNs with RNI are universal; random features break equivariance constraint enabling beyond-1-WL.

[17] [Spatio-Spectral Graph Neural Networks (Geisler et al., NeurIPS 2024)](https://arxiv.org/abs/2405.19121) — S²GNNs combining spatial and spectral filters; strictly more expressive than 1-WL.

[18] [The Expressive Power of Graph Neural Networks: A Survey (2024)](https://arxiv.org/html/2308.08235v2) — Comprehensive survey of GNN expressiveness covering WL hierarchy and aggregation function analysis.

[19] [Weisfeiler and Leman go Machine Learning (JMLR 2023)](https://jmlr.org/papers/volume24/22-0240/22-0240.pdf) — Survey connecting WL hierarchy to GNN expressiveness; failure cases on regular/SRG graphs.

## Follow-up Questions

- Can spectral heterodyning (multiplicative cross-scale interactions between eigenvector-weighted features at different diffusion timescales) provably exceed 1-WL expressiveness while remaining deterministic, or does equivariance still constrain such interactions?
- For non-regular graphs where NDS features do evolve, what is the precise characterization of which graph pairs NDS can vs. cannot distinguish — is there a clean algebraic characterization in terms of the graph spectrum?
- Could replacing ReLU with a non-monotone pointwise nonlinearity (e.g., sin, Gaussian) that maps some positive values to negative produce genuinely different fixed-point dynamics on regular graphs, or would the Perron eigenvector still dominate asymptotically?

---
*Generated by AI Inventor Pipeline*
