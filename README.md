# OTS-Bench: Optimal Transmission Switching for LLM Optimization Modeling

This repository contains the benchmark data, mathematical formulations, and solver code for the **Optimal Transmission Switching (OTS)** problem, contributed to the community-driven benchmark initiative for evaluating Large Language Models (LLMs) on optimization modeling (`OR-Bench`).

---

## 1. Problem Description

### Precise Technical Description
The Optimal Transmission Switching (OTS) problem aims to minimize the total generation cost in an electricity network by optimizing both generator outputs and network topology (line switching statuses). 

Given a network graph $\mathcal{G} = (\mathcal{N}, \mathcal{L})$ where $\mathcal{N}$ is the set of buses and $\mathcal{L}$ is the set of transmission lines:
* **Objective:** Minimize total active power generation cost.
* **Decision Variables:** Continuous generation outputs ($p_n$), continuous bus voltage angles ($\theta_n$), and binary line status variables ($x_l \in \{0,1\}$), where $x_l = 0$ indicates a line is switched off.
* **Constraints:** DC power flow equations, line thermal capacity limits and generation bounds. 

---

## 2. Mathematical Formulation

### 2.1 Notation

Consider a power system represented by a graph with nodes $\mathcal{N}$ and transmission lines $\mathcal{L}$. 

**Sets:**
- $\mathcal{N}$: Set of buses (nodes)
- $\mathcal{L}$: Set of transmission lines (arcs)

**Parameters:**
- $d_n$: Demand (load) at bus $n \in \mathcal{N}$ (MW)
- $c_n$: Marginal generation cost at bus $n \in \mathcal{N}$ (\$/MWh)
- $\underline{p}_n, \overline{p}_n$: Lower and upper generation limits at bus $n \in \mathcal{N}$ (MW)
- $b_l$: Susceptance of line $l \in \mathcal{L}$ (1/Ω)
- $\underline{f}_l, \overline{f}_l$: Lower and upper thermal capacity limits of line $l \in \mathcal{L}$ (MW)
- For line $l = (n, m)$: from-bus $n$ and to-bus $m$

**Decision Variables:**
- $p_n$: Active power generation at bus $n \in \mathcal{N}$ (MW, continuous)
- $\theta_n$: Bus voltage angle at bus $n \in \mathcal{N}$ (radians, continuous)
- $f_l$: Active power flow on line $l \in \mathcal{L}$ (MW, continuous)
- $\tilde{f}_l$: Dummy/unconstrained flow on line $l \in \mathcal{L}$ corresponding to $b_l(\theta_n - \theta_m)$ (MW, continuous)
- $x_l$: Binary status of line $l \in \mathcal{L}$ (1 = on, 0 = off)

---

### 2.2 Non-Linear Formulation (Conceptual)

The non-linear, mixed-integer DC-OTS problem can be conceptually formulated as:

$$
\begin{align}
\min_{p, f, \tilde{f}, \theta, x} \quad & \sum_{n \in \mathcal{N}} c_{n} \, p_{n} \\
\text{subject to} \quad & \nonumber \\ 
& f_l = x_l \tilde{f}_l, \quad \forall l \in \mathcal{L} \\
& \underline{f}_l \leq f_l \leq \overline{f}_l, \quad \forall l \in \mathcal{L} \\
& \tilde{f}_l = b_l(\theta_n-\theta_m), \quad \forall l=(n,m) \in \mathcal{L} \\
& p_n - d_n = \sum_{l\in\mathcal{L}(n,\cdot)} f_l - \sum_{l\in\mathcal{L}(\cdot,n)} f_l, \quad \forall n \in \mathcal{N} \\
& \underline{p}_n \leq p_n \leq \overline{p}_n, \quad \forall n \in \mathcal{N} \\
& \theta_1 = 0 \\
& x_l \in \{0,1\}, \quad \forall l \in \mathcal{L}
\end{align}
$$

**Key Challenge:** The constraint $f_l = x_l \tilde{f}_l$ is bilinear: when $x_l = 0$ (line off), it forces $f_l = 0$; when $x_l = 1$ (line on), it requires $f_l = \tilde{f}_l$. This product of a binary and continuous variable cannot be directly solved by standard MILP solvers.

---

### 2.3 Linearized MILP Formulation (Big-M Method)

To reformulate the problem as a Mixed-Integer Linear Program (MILP), we replace the bilinear constraint with linear Big-M disjunctive constraints. For each line $l$, we introduce constants $M_l^- < 0$ and $M_l^+ > 0$ that bound $\tilde{f}_l$ when the line is disconnected. The linearized formulation is:

$$
\begin{align}
\min_{p, f, \tilde{f}, \theta, x} \quad & \sum_{n \in \mathcal{N}} c_{n} \, p_{n} \\
\text{subject to} \quad & \nonumber \\ 
& (1-x_l)M_l^- \leq -f_l + \tilde{f}_l \leq (1-x_l)M_l^+, \quad \forall l \in \mathcal{L} \\
& x_l\underline{f}_l \leq f_l \leq x_l\overline{f}_l, \quad \forall l \in \mathcal{L} \\
& \tilde{f}_l = b_l(\theta_n-\theta_m), \quad \forall l=(n,m) \in \mathcal{L} \\
& p_n - d_n = \sum_{l\in\mathcal{L}(n,\cdot)} f_l - \sum_{l\in\mathcal{L}(\cdot,n)} f_l, \quad \forall n \in \mathcal{N} \\
& \underline{p}_n \leq p_n \leq \overline{p}_n, \quad \forall n \in \mathcal{N} \\
& \theta_1 = 0 \\
& x_l \in \{0,1\}, \quad \forall l \in \mathcal{L}
\end{align}
$$

**Linearization Logic:**
- When $x_l = 1$: $0 \leq -f_l + \tilde{f}_l \leq 0$, i.e., $f_l = \tilde{f}_l$ (coupling).
- When $x_l = 0$: $M_l^- \leq -f_l + \tilde{f}_l \leq M_l^+$, decoupling $f_l$ and $\tilde{f}_l$.
- The second constraint forces $f_l = 0$ when $x_l = 0$ (line is off), and applies thermal limits $\underline{f}_l \leq f_l \leq \overline{f}_l$ when $x_l = 1$.

The Big-M constants are computed internally based on the network topology to ensure numerical stability and correctness.


---

## 3. Repository Structure & Artifacts

The included IEEE 118-bus case is stored under
`power_system_cases/118_bus_system/`.

**power_system_cases/118_bus_system/buses.csv:** Contains bus data with columns:
- `BUS_ID`: Bus identifier
- `PD`: Demand (load) at the bus (MW)
- `PMIN`: Minimum generation capacity (MW)
- `PMAX`: Maximum generation capacity (MW)
- `COST`: Marginal generation cost (\$/MWh)

**power_system_cases/118_bus_system/branches.csv:** Contains transmission line data with columns:
- `F_BUS`: From-bus ID
- `T_BUS`: To-bus ID
- `BR_X`: Line reactance (Ω), used to compute susceptance $b_l = 1/BR_X$
- `RATE_A`: Thermal capacity limit (MW)

The Big-M parameters ($M_l^+$ and $M_l^-$) are computed internally during optimization based on the network topology:
1. For each line $l$, compute the maximum angle difference: $\Delta\theta_{max,l} = \frac{\text{RATEA}_l}{b_l}$
2. Sort these values and sum the top $N-1$ values (where $N$ is the number of buses), giving $\Delta\theta_{max}$
3. The Big-M constants are: $M_l^+ = \Delta\theta_{max} \cdot b_l$ and $M_l^- = -M_l^+$

This approach ensures numerically stable bounds based on actual network characteristics rather than arbitrary constants.

---

## 4. Getting Started

### Prerequisites
* Python 3.10+
* Gurobi Optimizer (with a valid license installed)

### Installation

```bash
git clone https://github.com/salvapineda/ots_bench.git
cd ots_bench
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Solver

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python3 ots.py
```

The solver will output the minimum cost and the list of transmission lines switched off.

---

## 5. Contributors

- **Salvador Pineda** (spineda@uma.es) - University of Málaga
- **Juan Miguel Morales** (juan.morales@uma.es) - University of Málaga
