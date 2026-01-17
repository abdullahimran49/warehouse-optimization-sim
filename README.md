# Warehouse Optimization Simulation 🏭

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

A Python-based simulation that optimizes warehouse workforce configuration and capacity planning. This project utilizes and compares **Hill Climbing** and **Genetic Algorithms** against a baseline configuration to maximize throughput and minimize operational costs.

## 📋 Table of Contents
- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [License](#-license)

## 📊 Overview

Managing a warehouse involves balancing conflicting goals: maximizing order processing speed while minimizing worker idle time and travel distance. This simulation models a warehouse environment with specific constraints (picking, packing, loading) and uses AI optimization techniques to find the ideal setup.

The simulation attempts to optimize parameters such as:
* Number of **Acceptors, Controllers, Loaders, and Forklifts**.
* Capacity of **Queues, Placement Zones, and Dispatch Zones**.

## 🧠 How It Works

The simulation evaluates a "Fitness Score" (lower is better) for every configuration based on a weighted sum of four key metrics:

$$Fitness = (0.4 \times AvgTime) + (0.3 \times (100 - Util\%)) + (0.2 \times Distance) + (0.1 \times IdleTime)$$

### Algorithms Implemented
1.  **Baseline Simulation:** Runs the warehouse with standard/default parameters.
2.  **Hill Climbing:** A local search algorithm that iteratively tweaks parameters to find a local optimum.
3.  **Genetic Algorithm:** An evolutionary approach that uses selection, crossover, and mutation to evolve a population of solutions over generations.

## ✨ Features

* **Discrete Event Simulation:** Models stochastic order arrivals and processing times.
* **Multi-Objective Optimization:** Balances speed, cost, and efficiency.
* **Visual Analytics:** `matplotlib` integration generates comprehensive charts comparing convergence and performance metrics.
* **Detailed Reporting:** Console output provides a granular breakdown of Orders Per Minute (OPM) and improvement percentages.

## 💻 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/abdullah_imran49/warehouse-optimization-sim.git](https://github.com/abdullah_imran49/warehouse-optimization-sim.git)
    cd warehouse-optimization-sim
    ```

2.  **Install dependencies**
    This project requires `numpy` for calculation and `matplotlib` for visualization.
    ```bash
    pip install numpy matplotlib
    ```

## 🚀 Usage

Run the main simulation script:

```bash
python warehouse_sim.py
