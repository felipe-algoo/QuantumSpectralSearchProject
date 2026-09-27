### `Overview`

- The framework implements benchmark graph families, adjacency andLaplacian spectral analysis, CTQW search dynamics with optional Hermitiannoise, statistical inference.
  
### `Installation`

- Linux
  
        python3.10 -m venv .venvsource .venv/bin/activatepip install -r requirements.txtpip install -e

- Windows

        py -3.10 -m venv .venv.\.venv\Scripts\Activate.ps1pip install -r requirements.txtpip install -e 

### `Configuration`

- All experimental parameters live in ***config/parameters.yaml.*** Random seeds arederived deterministically from the base_seed field and a hash of the ***(graph_type, n_nodes, instance_idx, noise_level)*** tuple.

### `Running experiments`

- results/experiment_summary.csv: flat summary table.
- results/experiment_full_records.json: full per run records includingeigenvalue spectra and full probability time series.
- results/experiment_config.json: frozen configuration used.

### `Tests`

- Tests validate graph generation, spectral decomposition against closed-formeigenvalues of complete/path/cycle graphs, 
- Hamiltonian Hermiticity, unitarityof evolution, and the analytical success probability of CTQW search on thecomplete graph K n(expected p ∗ →1 at t ∗=π n/2).

### `Methodology`


- The continuous-time quantum walk search Hamiltonian is H=−γA−∣w⟩⟨w∣, where A is the graph adjacency matrix, γ is the jumping rate, and ∣w⟩ is the marked vertex oracle. 
- The system is initialised in theuniform superposition ∣s⟩= N 1∑ v∣v⟩, and evolves as ∣ψ(t)⟩=e −iHt ∣s⟩. The success probabilityis p w(t)=∣⟨w∣ψ(t)⟩∣ 2.

- The framework systematically explores how the adjacency spectral gap Δ=λ 1(A)−λ 2(A) and the Laplacian algebraic connectivity λ 
- 2(L) correlate with max tp w(t) and the optimal hitting time t ∗=argmax tp w(t) acrossfive graph families, five graph sizes, five noise levels, and five instancesper configuration.

- Graph families

        Path	networkx.path_graph
        Cycle	networkx.cycle_graph
        Complete	networkx.complete_graph
        Erdős–Rényi	networkx.gnp_random_graph(n, p, seed)
        Barabási–Albert	networkx.barabasi_albert_graph(n, m, seed)

- Spectral decomposition

- For each graph we compute the full eigen-decomposition of the adjacencymatrix A and the Laplacian L=D−A via numpy.linalg.eigh (dense) orscipy.sparse.linalg.eigsh (sparse, k<n). 
- The adjacency spectral gap is ΔA=λ 1(A)−λ 2(A), and the Laplacian algebraic connectivity is λ 2(L) (the smallestnon-zero eigenvalue).

- Quantum walk search

- The Childs Goldstone search Hamiltonian is H=−γA−∣w⟩⟨w∣. For d$-regular graphs the optimal jumping rate is $\gamma = 1/d. We supportthree γ strategies:

          average_degree: γ=1/ kˉwhere  kˉis the mean degree.
          spectral_gap: γ=1/Δ A.
          uniform: γ=1/N.

- Time evolution uses scipy.sparse.linalg.expm_multiply, which applies aKrylov-subspace approximation of e −iHt∣s⟩.

- Noise model

- We perturb the Hamiltonian with a Gaussian Hermitian operator Hσ=H+σ⋅ ∥W∥ 2W,W=2G+G⊤,Gij∼N(0,1). This preserves Hermiticity and bounds the perturbation by σ inoperator norm.

- Statistical analysis

          Bootstrap confidence intervals: 10,000 resamples, 95% level.
          OLS regression: maxtpw(t)∼β 0+β1ΔA.
          Hypothesis test: one-sided Pearson correlation H1:ρ>0,α=0.05.
          Effect size: Cohen's d between noise-free and noisy regimes.
