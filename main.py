from initialization_module import create_scenario_1, create_scenario_2
from simulation_engine_module import SimulationEngine
from visualization_module import SimulationVisualizer

def main():
    """Main entry point for UCLF simulation"""
    
    print("\n" + "="*80)
    print("UCLF: Uncertainty-Aware Cooperative Lane-Changing Framework")
    print("Simulation System for Connected Autonomous Vehicles")
    print("="*80 + "\n")
    
    # Choose scenario
    print("Available Scenarios:")
    print("1. Lane Blockage Scenario")
    print("2. Lane Drop Scenario")
    choice = input("\nSelect scenario (1 or 2): ")
    
    if choice == "2":
        scenario = create_scenario_2()
    else:
        scenario = create_scenario_1()
    
    # Print scenario summary
    scenario.print_scenario_summary()
    
    # Create simulation engine
    engine = SimulationEngine(scenario)
    
    # Run simulation
    duration = float(input("Enter simulation duration in seconds (default 10): ") or "10")
    
    print("\nRunning simulation...")
    engine.run_simulation(duration=duration, verbose=True)
    
    # Print summary
    summary = engine.get_summary()
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Total Simulation Time: {summary['total_time']:.2f}s")
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Total Collision Events: {summary['total_collisions']}")
    print("="*80 + "\n")
    
    # Optional: Visualization
    visualize = input("Generate visualization? (y/n): ")
    if visualize.lower() == 'y':
        try:
            visualizer = SimulationVisualizer(engine.state_manager)
            visualizer.plot_vehicle_trajectories(scenario.obstacles)
            visualizer.plot_velocity_profiles()
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
    
    # Optional: Export to CSV
    export = input("Export data to CSV? (y/n): ")
    if export.lower() == 'y':
        visualizer = SimulationVisualizer(engine.state_manager)
        visualizer.export_to_csv('uclf_simulation_results.csv')

if __name__ == "__main__":
    main()
