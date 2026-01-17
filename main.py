import numpy as np
import random
import matplotlib.pyplot as plt

# Warehouse parameters
params = {
    'acceptorsNumber': 2, 'controllersNumber': 2, 'loadersNumber': 2,
    'unloadersNumber': 2, 'transferersNumber': 2, 'forkliftNumber': 3,
    'orderQueueCapacity': 10, 'placementZoneCapacity': 20,
    'dispatchZoneCapacity': 8, 'orderInterarrivalTimeMin': 3,
    'orderInterarrivalTimeMax': 5, 'simulation_time': 480
}

def simulate(p):
    """Run warehouse simulation and return metrics"""
    # Order processing
    picking_rate = p['acceptorsNumber'] * 2.5
    packing_rate = p['controllersNumber'] * 2.0
    loading_rate = p['loadersNumber'] * 3.0
    service_rate = min(picking_rate, packing_rate, loading_rate)
    
    arrival_rate = 60 / ((p['orderInterarrivalTimeMin'] + p['orderInterarrivalTimeMax']) / 2)
    utilization = min(0.95, arrival_rate / service_rate)
    
    avg_completion = (1 / service_rate) * (1 / (1 - utilization))
    
    # Travel distance
    total_capacity = p['placementZoneCapacity'] + p['dispatchZoneCapacity']
    base_distance = np.sqrt(total_capacity) * 10
    efficiency = min(1.0, p['forkliftNumber'] / 3.0)
    avg_distance = base_distance / efficiency
    
    # Worker utilization
    total_workers = (p['acceptorsNumber'] + p['controllersNumber'] + p['loadersNumber'] +
                    p['unloadersNumber'] + p['transferersNumber'] + p['forkliftNumber'])
    work_per_order = 15
    total_work = arrival_rate * work_per_order * p['simulation_time']
    available_capacity = total_workers * p['simulation_time']
    worker_util = min(100, (total_work / available_capacity) * 100)
    
    # Throughput
    throughput = min(p['simulation_time'] / avg_completion, p['orderQueueCapacity'] * 10)
    
    # Idle time
    idle_time = (100 - worker_util) / 10
    
    # Fitness score (lower is better)
    fitness = (avg_completion * 0.4 + (100 - worker_util) * 0.3 + 
              avg_distance * 0.2 + idle_time * 0.1)
    
    # Orders per minute
    orders_per_min = throughput / p['simulation_time']
    
    return {
        'avg_completion': avg_completion, 'throughput': throughput,
        'worker_util': worker_util, 'avg_distance': avg_distance,
        'idle_time': idle_time, 'fitness': fitness,
        'orders_per_min': orders_per_min, 'orders_processed': int(throughput)
    }

def get_neighbor(p):
    """Generate neighbor solution"""
    neighbor = p.copy()
    param_choices = ['acceptorsNumber', 'controllersNumber', 'loadersNumber',
                    'forkliftNumber', 'orderQueueCapacity', 'unloadersNumber',
                    'transferersNumber', 'placementZoneCapacity']
    
    param = random.choice(param_choices)
    current = neighbor[param]
    change = random.choice([-1, 1])
    new_value = max(1, current + change)
    
    if param == 'orderQueueCapacity':
        new_value = min(20, new_value)
    elif 'Number' in param:
        new_value = min(10, new_value)
    elif 'Capacity' in param:
        new_value = min(30, new_value)
    
    neighbor[param] = new_value
    return neighbor

def hill_climbing(base_params, max_iter=100):
    """Hill climbing optimization"""
    print("\n=== Hill Climbing Optimization ===")
    current = base_params.copy()
    current_metrics = simulate(current)
    current_fitness = current_metrics['fitness']
    
    best = current.copy()
    best_fitness = current_fitness
    history = []
    
    for i in range(max_iter):
        neighbor = get_neighbor(current)
        neighbor_metrics = simulate(neighbor)
        neighbor_fitness = neighbor_metrics['fitness']
        
        if neighbor_fitness < current_fitness:
            current = neighbor
            current_fitness = neighbor_fitness
            
            if neighbor_fitness < best_fitness:
                best = neighbor.copy()
                best_fitness = neighbor_fitness
                print(f"Iteration {i}: New best fitness = {best_fitness:.2f}")
        
        history.append(current_fitness)
    
    return best, simulate(best), history

def genetic_algorithm(base_params, pop_size=30, generations=50):
    """Genetic algorithm optimization"""
    print("\n=== Genetic Algorithm Optimization ===")
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        ind = base_params.copy()
        ind['acceptorsNumber'] = random.randint(1, 6)
        ind['controllersNumber'] = random.randint(1, 5)
        ind['loadersNumber'] = random.randint(1, 5)
        ind['forkliftNumber'] = random.randint(2, 6)
        ind['orderQueueCapacity'] = random.randint(5, 15)
        ind['unloadersNumber'] = random.randint(1, 5)
        ind['transferersNumber'] = random.randint(1, 5)
        ind['placementZoneCapacity'] = random.randint(15, 30)
        population.append(ind)
    
    best = None
    best_fitness = float('inf')
    history = []
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [simulate(ind)['fitness'] for ind in population]
        
        # Track best
        min_idx = np.argmin(fitness_scores)
        if fitness_scores[min_idx] < best_fitness:
            best_fitness = fitness_scores[min_idx]
            best = population[min_idx].copy()
            print(f"Generation {gen}: Best fitness = {best_fitness:.2f}")
        
        history.append(min(fitness_scores))
        
        # Selection (tournament)
        selected = []
        for _ in range(len(population) // 2):
            indices = random.sample(range(len(population)), 3)
            tournament = [(population[i], fitness_scores[i]) for i in indices]
            winner = min(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        
        # New population with elitism
        new_pop = [population[min_idx].copy()]
        
        # Crossover and mutation
        param_keys = ['acceptorsNumber', 'controllersNumber', 'loadersNumber',
                     'forkliftNumber', 'orderQueueCapacity', 'unloadersNumber']
        
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = p1.copy()
            
            # Crossover
            crossover_point = random.randint(0, len(param_keys))
            for i, k in enumerate(param_keys):
                if i >= crossover_point:
                    child[k] = p2[k]
            
            # Mutation
            if random.random() < 0.2:
                param = random.choice(param_keys)
                current = child[param]
                change = random.choice([-1, 0, 1])
                new_value = max(1, current + change)
                if 'Number' in param:
                    new_value = min(8, new_value)
                elif param == 'orderQueueCapacity':
                    new_value = min(20, new_value)
                child[param] = new_value
            
            new_pop.append(child)
        
        population = new_pop
    
    return best, simulate(best), history

def print_params(p, title):
    """Print parameters"""
    print(f"\n{'='*60}\n{title:^60}\n{'='*60}")
    print(f"Workforce: Acceptors={p['acceptorsNumber']}, Controllers={p['controllersNumber']}, "
          f"Loaders={p['loadersNumber']}, Unloaders={p['unloadersNumber']}, "
          f"Transferers={p['transferersNumber']}, Forklifts={p['forkliftNumber']}")
    print(f"Capacity: Queue={p['orderQueueCapacity']}, Placement={p['placementZoneCapacity']}, "
          f"Dispatch={p['dispatchZoneCapacity']}")

def print_metrics(m, title):
    """Print metrics"""
    print(f"\n{title}\n{'-'*60}")
    print(f"Avg Order Completion: {m['avg_completion']:.2f} min | Throughput: {m['throughput']:.2f} orders")
    print(f"Orders Processed: {m['orders_processed']} | Worker Util: {m['worker_util']:.2f}%")
    print(f"Avg Travel Distance: {m['avg_distance']:.2f} m | Idle Time: {m['idle_time']:.2f}")
    print(f"Orders/Minute: {m['orders_per_min']:.4f} | Fitness: {m['fitness']:.2f}")

def plot_results(base_m, hc_m, ga_m, hc_hist, ga_hist):
    """Create all visualizations"""
    fig = plt.figure(figsize=(16, 10))
    
    # Convergence plots
    plt.subplot(2, 3, 1)
    plt.plot(hc_hist, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Score')
    plt.title('Hill Climbing Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(ga_hist, 'r-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Genetic Algorithm Convergence')
    plt.grid(True, alpha=0.3)
    
    # Base case visualization
    plt.subplot(2, 3, 3)
    metrics = ['Completion\nTime', 'Throughput', 'Worker\nUtil %', 'Travel\nDist']
    values = [base_m['avg_completion'], base_m['throughput'], 
             base_m['worker_util'], base_m['avg_distance']]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Base Case Metrics')
    plt.ylabel('Value')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Comparison bar chart
    plt.subplot(2, 3, 4)
    methods = ['Base', 'Hill Climb', 'Genetic']
    completion_times = [base_m['avg_completion'], hc_m['avg_completion'], ga_m['avg_completion']]
    x = np.arange(len(methods))
    bars = plt.bar(x, completion_times, color=['#95a5a6', '#3498db', '#e74c3c'], 
                   alpha=0.7, edgecolor='black')
    plt.ylabel('Minutes')
    plt.title('Avg Completion Time Comparison')
    plt.xticks(x, methods)
    for bar, val in zip(bars, completion_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(completion_times)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Throughput comparison
    plt.subplot(2, 3, 5)
    throughputs = [base_m['throughput'], hc_m['throughput'], ga_m['throughput']]
    bars = plt.bar(x, throughputs, color=['#95a5a6', '#3498db', '#e74c3c'], 
                   alpha=0.7, edgecolor='black')
    plt.ylabel('Orders')
    plt.title('Throughput Comparison')
    plt.xticks(x, methods)
    for bar, val in zip(bars, throughputs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Orders per minute comparison
    plt.subplot(2, 3, 6)
    orders_per_min = [base_m['orders_per_min'], hc_m['orders_per_min'], ga_m['orders_per_min']]
    bars = plt.bar(x, orders_per_min, color=['#95a5a6', '#3498db', '#e74c3c'], 
                   alpha=0.7, edgecolor='black')
    plt.ylabel('Orders/Minute')
    plt.title('Orders Per Minute Comparison')
    plt.xticks(x, methods)
    for bar, val in zip(bars, orders_per_min):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(orders_per_min)*0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    print("\nVisualization displayed")

def print_comparison_table(base_m, hc_m, ga_m):
    """Print detailed comparison table"""
    print(f"\n{'='*90}")
    print(f"{'ORDERS PER MINUTE ANALYSIS':^90}")
    print(f"{'='*90}")
    print(f"{'Method':<20} {'Orders/Min':<20} {'Total Orders':<20} {'Improvement':<20}")
    print(f"{'-'*90}")
    
    base_opm = base_m['orders_per_min']
    hc_opm = hc_m['orders_per_min']
    ga_opm = ga_m['orders_per_min']
    
    hc_imp = ((hc_opm - base_opm) / base_opm) * 100
    ga_imp = ((ga_opm - base_opm) / base_opm) * 100
    
    print(f"{'Base Case':<20} {base_opm:<20.6f} {base_m['orders_processed']:<20} {'-':<20}")
    print(f"{'Hill Climbing':<20} {hc_opm:<20.6f} {hc_m['orders_processed']:<20} {f'+{hc_imp:.2f}%':<20}")
    print(f"{'Genetic Algorithm':<20} {ga_opm:<20.6f} {ga_m['orders_processed']:<20} {f'+{ga_imp:.2f}%':<20}")
    
    print(f"\n{'='*90}")
    print(f"{'COMPREHENSIVE METRICS COMPARISON':^90}")
    print(f"{'='*90}")
    print(f"{'Metric':<35} {'Base':<15} {'Hill Climb':<15} {'Genetic':<15}")
    print(f"{'-'*90}")
    
    metrics_data = [
        ('Avg Completion Time (min)', base_m['avg_completion'], hc_m['avg_completion'], ga_m['avg_completion']),
        ('Total Throughput (orders)', base_m['throughput'], hc_m['throughput'], ga_m['throughput']),
        ('Worker Utilization (%)', base_m['worker_util'], hc_m['worker_util'], ga_m['worker_util']),
        ('Avg Travel Distance (m)', base_m['avg_distance'], hc_m['avg_distance'], ga_m['avg_distance']),
        ('Fitness Score', base_m['fitness'], hc_m['fitness'], ga_m['fitness'])
    ]
    
    for name, base, hc, ga in metrics_data:
        print(f"{name:<35} {base:<15.2f} {hc:<15.2f} {ga:<15.2f}")
    
    print(f"\n{'='*90}")
    print(f"{'PERFORMANCE IMPROVEMENT vs BASE CASE':^90}")
    print(f"{'='*90}")
    print(f"{'Metric':<35} {'Hill Climb':<25} {'Genetic':<25}")
    print(f"{'-'*90}")
    
    for i, (name, base, hc, ga) in enumerate(metrics_data):
        if i in [1, 2]:  # Throughput and utilization (higher is better)
            hc_imp = ((hc - base) / base) * 100
            ga_imp = ((ga - base) / base) * 100
        else:  # Other metrics (lower is better)
            hc_imp = ((base - hc) / base) * 100
            ga_imp = ((base - ga) / base) * 100
        
        print(f"{name:<35} {f'+{hc_imp:.2f}%':<25} {f'+{ga_imp:.2f}%':<25}")

# Main execution
print("="*90)
print("WAREHOUSE OPTIMIZATION SIMULATION")
print("="*90)

# Base case
print("\n" + "="*90)
print("BASE CASE SIMULATION")
print("="*90)
print_params(params, "Base Case Parameters")
base_metrics = simulate(params)
print_metrics(base_metrics, "Base Case Performance")

# Hill Climbing
print("\n" + "="*90)
print("HILL CLIMBING OPTIMIZATION")
print("="*90)
hc_best, hc_metrics, hc_history = hill_climbing(params, 100)
print_params(hc_best, "Hill Climbing Optimized Parameters")
print_metrics(hc_metrics, "Hill Climbing Performance")

# Genetic Algorithm
print("\n" + "="*90)
print("GENETIC ALGORITHM OPTIMIZATION")
print("="*90)
ga_best, ga_metrics, ga_history = genetic_algorithm(params, 30, 50)
print_params(ga_best, "Genetic Algorithm Optimized Parameters")
print_metrics(ga_metrics, "Genetic Algorithm Performance")

# Comparison
print_comparison_table(base_metrics, hc_metrics, ga_metrics)

# Visualization
plot_results(base_metrics, hc_metrics, ga_metrics, hc_history, ga_history)

print("\n" + "="*90)
print("SIMULATION COMPLETE")
print("="*90)
