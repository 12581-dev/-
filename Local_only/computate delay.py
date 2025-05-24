import re
import numpy as np
import matplotlib.pyplot as plt

def analyze_output_file(file_path='output.txt'):
    """
    Analyzes the output.txt file from the UAV simulation.
    Extracts delay information for each episode and creates visualizations.
    """
    # Read the file
    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
    # Split content by episodes
    episodes = content.split("======== This episode is done ========")
    
    # Initialize list to store episode delays
    episode_delays = []
    
    # Regular expression to extract delay values
    delay_pattern = re.compile(r"delay:([\d.]+)")
    
    # Process each episode
    for i, episode in enumerate(episodes):
        if not episode.strip():
            continue
        
        # Extract all delay values in this episode
        delays = delay_pattern.findall(episode)
        
        # Convert to float and sum
        if delays:
            total_delay = sum(float(d) for d in delays)
            episode_delays.append(total_delay)
            print(f"Episode {i}: Total delay = {total_delay:.2f}")
    
    return episode_delays

def visualize_results(episode_delays):
    """
    Creates visualizations for the extracted delay data.
    """
    if not episode_delays:
        print("No data to visualize.")
        return
    
    # Convert to numpy array
    delays = np.array(episode_delays)
    episodes = np.arange(1, len(delays) + 1)
    
    # Basic statistics
    print("\nStatistics:")
    print(f"Average delay per episode: {np.mean(delays):.2f}")
    print(f"Minimum delay: {np.min(delays):.2f}")
    print(f"Maximum delay: {np.max(delays):.2f}")
    print(f"Standard deviation: {np.std(delays):.2f}")
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Line plot of delay vs episode
    plt.subplot(2, 1, 1)
    plt.plot(episodes, delays, marker='o', linestyle='-', color='blue')
    plt.title('Total Delay per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Delay (s)')
    plt.grid(True)
    
    # Plot 2: Moving average to show trend
    window_size = min(10, len(delays))
    if window_size > 1:
        # Calculate moving average manually
        moving_avg = []
        for i in range(len(delays)):
            if i < window_size - 1:
                # Not enough previous points for full window
                moving_avg.append(np.mean(delays[:i+1]))
            else:
                moving_avg.append(np.mean(delays[i-window_size+1:i+1]))
        
        plt.subplot(2, 1, 2)
        plt.plot(episodes, delays, alpha=0.3, color='blue', label='Raw Data')
        plt.plot(episodes, moving_avg, linewidth=2, color='red', 
                label=f'{window_size}-Episode Moving Average')
        plt.title(f'Delay Trend with {window_size}-Episode Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Delay (s)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("delay_analysis.png", dpi=300)
    plt.show()
    
    # Additional visualization: Histogram of delays
    plt.figure(figsize=(10, 6))
    plt.hist(delays, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Episode Delays')
    plt.xlabel('Total Delay (s)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig("delay_histogram.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Analyzing UAV simulation output file...")
    episode_delays = analyze_output_file()
    
    if episode_delays is not None:
        print(f"\nFound data for {len(episode_delays)} episodes.")
        visualize_results(episode_delays)
    else:
        print("Analysis failed. Please check if the output.txt file exists and has the correct format.")