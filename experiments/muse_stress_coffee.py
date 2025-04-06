#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime
import threading
import argparse

# Import for Muse-LSL integration
try:
    from muselsl import stream, list_muses, view, record
    from pylsl import StreamInlet, resolve_byprop
    MUSE_LSL_AVAILABLE = True
except ImportError:
    print("Warning: muselsl not found. Will use simulated data.")
    MUSE_LSL_AVAILABLE = False

# Import from our coffee recommendation system
from stress_coffee_recommendation import CoffeeRecommender, StressAnalyzer

"""
Stress-Based Coffee Recommendation System Experiment with Muse Headband
- Uses real EEG data from Muse 2 headband via Muse-LSL
- Calculates stress levels from EEG frequency bands
- Makes personalized coffee recommendations based on measured stress
"""

class MuseLSLCollector:
    """
    Class for collecting EEG data from a real Muse headband using Muse-LSL
    """
    def __init__(self):
        self.connected = False
        self.streaming = False
        self.inlet = None
        self.stream_thread = None
        self.recording = False
        self.record_thread = None
        self.data_buffer = []
        self.channels = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX']
        
    def list_available_devices(self):
        """List available Muse devices"""
        if not MUSE_LSL_AVAILABLE:
            print("Error: muselsl library not available")
            return []
        
        print("Searching for Muse devices...")
        muses = list_muses()
        
        if not muses:
            print("No Muse devices found.")
        else:
            print(f"Found {len(muses)} Muse device(s):")
            for i, muse in enumerate(muses):
                print(f"[{i}] Name: {muse['name']}, MAC Address: {muse['address']}")
                
        return muses
    
    def connect_and_stream(self, address=None, name=None):
        """Connect to a Muse device and start streaming"""
        if not MUSE_LSL_AVAILABLE:
            print("Error: muselsl library not available")
            return False
        
        if self.streaming:
            print("Already streaming. Stop current stream first.")
            return False
        
        try:
            # Start streaming in a separate thread to not block execution
            def stream_starter():
                try:
                    if address:
                        print(f"Connecting to Muse at {address}...")
                        stream(address=address)
                    elif name:
                        print(f"Connecting to Muse with name {name}...")
                        muses = list_muses()
                        found = False
                        for muse in muses:
                            if muse['name'] == name:
                                stream(address=muse['address'])
                                found = True
                                break
                        if not found:
                            print(f"No Muse found with name {name}")
                            return
                    else:
                        print("Connecting to first available Muse...")
                        stream()
                except Exception as e:
                    print(f"Error connecting to Muse: {e}")
                    self.streaming = False
            
            self.stream_thread = threading.Thread(target=stream_starter)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            # Wait a moment for streaming to start
            time.sleep(3)
            
            # Check if stream is available and set up inlet
            print("Looking for an EEG stream...")
            streams = resolve_byprop('type', 'EEG', timeout=5)
            
            if len(streams) == 0:
                print("No EEG stream found. Make sure your device is connected properly.")
                self.streaming = False
                return False
                
            # Create a new inlet to read from the stream
            self.inlet = StreamInlet(streams[0])
            self.streaming = True
            self.connected = True
            print("Successfully connected to Muse EEG stream")
            return True
            
        except Exception as e:
            print(f"Error connecting to Muse: {e}")
            self.streaming = False
            self.connected = False
            return False
    
    def record_data(self, duration=60, filename=None):
        """Record EEG data to a file"""
        if not MUSE_LSL_AVAILABLE:
            print("Error: muselsl library not available")
            return False
        
        if not self.streaming:
            print("Not streaming. Start streaming before recording.")
            return False
        
        if self.recording:
            print("Already recording. Stop current recording first.")
            return False
        
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"muse_eeg_{timestamp}"
            
            # Start recording in a separate thread
            def record_starter():
                try:
                    print(f"Recording EEG data for {duration} seconds...")
                    record(duration=duration, filename=filename)
                    print(f"Recording complete. Saved to {filename}.csv")
                    self.recording = False
                except Exception as e:
                    print(f"Error recording data: {e}")
                    self.recording = False
            
            self.record_thread = threading.Thread(target=record_starter)
            self.record_thread.daemon = True
            self.record_thread.start()
            self.recording = True
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.recording = False
            return False
    
    def collect_samples(self, duration=5.0, sampling_rate=256):
        """Collect EEG samples for a specified duration"""
        if not self.streaming or self.inlet is None:
            print("Not streaming or inlet not available")
            return None
        
        print(f"Collecting {duration} seconds of EEG data...")
        
        # Calculate number of samples to collect
        n_samples = int(duration * sampling_rate)
        
        # Initialize data buffer
        self.data_buffer = []
        
        # Collect samples
        start_time = time.time()
        collected = 0
        
        while collected < n_samples and (time.time() - start_time) < duration + 1:
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                if sample is not None:
                    self.data_buffer.append((sample, timestamp))
                    collected += 1
            except Exception as e:
                print(f"Error collecting samples: {e}")
                break
        
        print(f"Collected {len(self.data_buffer)} samples")
        
        if len(self.data_buffer) == 0:
            return None
            
        return self.data_buffer
    
    def disconnect(self):
        """Disconnect from the Muse headband"""
        self.streaming = False
        self.connected = False
        self.inlet = None
        print("Disconnected from Muse")
        return True
    
    def process_to_frequency_bands(self, data_buffer=None):
        """
        Process raw EEG data into frequency bands
        This is a simplified example - a real implementation would use proper 
        signal processing techniques like FFT or wavelet transform
        """
        if data_buffer is None:
            data_buffer = self.data_buffer
            
        if not data_buffer:
            print("No data available to process")
            return None
            
        # Extract just the EEG data (not timestamps)
        eeg_data = np.array([sample[0] for sample in data_buffer])
        
        # In real implementation, we would do proper frequency analysis
        # For now, let's simulate it with a crude approach
        
        # Create a dictionary for the frequency bands
        bands = {
            'delta': (1, 4),    # 1-4 Hz
            'theta': (4, 8),    # 4-8 Hz
            'alpha': (8, 13),   # 8-13 Hz
            'beta': (13, 30),   # 13-30 Hz
            'gamma': (30, 50)   # 30-50 Hz
        }
        
        # Create empty dictionary for the processed data
        processed_data = {}
        
        # For each band, simulate band power
        # In a real implementation, we would use FFT or wavelet transform
        for band_name, (low_freq, high_freq) in bands.items():
            # Calculate approximate amplitude in the frequency range
            # This is just a simulation, not actual frequency analysis
            
            # For demonstration, we'll use different channels in different ways
            if band_name == 'alpha':
                # Alpha is typically higher in relaxed states
                # Use frontal channels (AF7, AF8) for alpha
                channel_idx = [1, 2]  # AF7, AF8
            elif band_name == 'beta':
                # Beta is typically higher during active, busy, or anxious thinking
                # Use all channels
                channel_idx = [0, 1, 2, 3]  # All channels
            else:
                # Use all channels for other bands
                channel_idx = [0, 1, 2, 3]
                
            # Extract relevant channels
            channel_data = eeg_data[:, channel_idx]
            
            # Calculate a simulated band power
            # In real implementation, this would be proper spectral analysis
            band_power = np.mean(np.abs(channel_data))
            
            # Add random variations to simulate the band
            if band_name == 'alpha':
                # Alpha should be higher when relaxed (low stress)
                band_power *= np.random.uniform(0.8, 1.2)
            elif band_name == 'beta':
                # Beta should be higher when stressed
                band_power *= np.random.uniform(0.9, 1.4)
            
            processed_data[band_name] = band_power
            
        return processed_data


class StressEEGAnalyzer(StressAnalyzer):
    """
    Extended stress analyzer that can work with real EEG data
    """
    def __init__(self):
        super().__init__()
        
        # EEG-specific parameters for stress calculation
        self.alpha_beta_weight = 0.6  # Weight for alpha/beta ratio
        self.theta_weight = 0.2       # Weight for theta power (meditation)
        self.delta_weight = 0.1       # Weight for delta power (deep sleep)
        self.gamma_weight = 0.1       # Weight for gamma (cognitive processing)
    
    def calculate_stress_from_bands(self, band_powers):
        """
        Calculate stress level from frequency band powers
        
        This uses established neurological correlates of stress:
        - Decreased alpha and increased beta power are associated with stress
        - Higher theta can indicate meditative states (lower stress)
        - Gamma activity can be associated with cognitive processing
        
        References:
        - W. Klimesch, "EEG alpha and theta oscillations reflect cognitive and 
          memory performance: a review and analysis," Brain Research Reviews, 1999
        - J.B. Nitschke et al., "Anxiety disorder patients show reduced activity in
          alpha-band EEG," Biological Psychiatry, 1999
        """
        if not band_powers:
            return None
            
        # Calculate the alpha/beta ratio (key stress indicator)
        # Lower values typically indicate higher stress
        alpha = band_powers.get('alpha', 0)
        beta = band_powers.get('beta', 0)
        theta = band_powers.get('theta', 0)
        delta = band_powers.get('delta', 0)
        gamma = band_powers.get('gamma', 0)
        
        # Avoid division by zero
        if beta == 0:
            beta = 0.001
            
        # Alpha/Beta ratio - main stress indicator
        alpha_beta_ratio = alpha / beta
        
        # Normalize to a 0-1 scale (will be inverted for stress)
        # Typical alpha/beta ratios range from 0.5 (stressed) to 2.5 (relaxed)
        normalized_ratio = min(1.0, max(0.0, (alpha_beta_ratio - 0.5) / 2.0))
        
        # Theta component - higher theta can indicate meditation/relaxation
        # but also drowsiness
        theta_component = min(1.0, max(0.0, theta / (alpha + 0.001)))
        
        # Gamma component - high gamma with high beta can indicate stress/anxiety
        gamma_component = min(1.0, max(0.0, gamma / (alpha + 0.001)))
        
        # Calculate relaxation score (0-1, where 1 is very relaxed)
        relaxation_score = (
            self.alpha_beta_weight * normalized_ratio +
            self.theta_weight * theta_component +
            self.delta_weight * (delta / (beta + 0.001)) -
            self.gamma_weight * gamma_component
        )
        
        # Clamp between 0-1
        relaxation_score = min(1.0, max(0.0, relaxation_score))
        
        # Convert to stress score (0-100, where 100 is very stressed)
        stress_level = 100 * (1.0 - relaxation_score)
        
        # Record the stress level
        self.current_stress = stress_level
        self.stress_history.append((datetime.now(), stress_level))
        
        return stress_level


class MuseStressExperiment:
    """
    Run the stress-based coffee recommendation experiment with a Muse headband
    """
    def __init__(self, use_real_muse=True):
        self.use_real_muse = use_real_muse and MUSE_LSL_AVAILABLE
        
        if self.use_real_muse:
            self.muse_collector = MuseLSLCollector()
        else:
            # Fallback to simulated data collector
            from stress_coffee_recommendation import MuseDataCollector
            self.muse_collector = MuseDataCollector()
            
        self.stress_analyzer = StressEEGAnalyzer()
        self.coffee_recommender = CoffeeRecommender()
    
    def setup(self):
        """Setup the experiment"""
        if self.use_real_muse:
            # List available devices
            muses = self.muse_collector.list_available_devices()
            
            if muses:
                # Connect to the first available device
                return self.muse_collector.connect_and_stream()
            else:
                print("No Muse devices found. Switching to simulation mode.")
                self.use_real_muse = False
                from stress_coffee_recommendation import MuseDataCollector
                self.muse_collector = MuseDataCollector()
                return self.muse_collector.connect()
        else:
            return self.muse_collector.connect()
    
    def measure_stress(self, duration=10):
        """Measure stress levels from EEG data"""
        print(f"\n=== Measuring Stress Levels (Duration: {duration}s) ===")
        
        # Collect data
        if self.use_real_muse:
            # Collect real EEG data
            eeg_data = self.muse_collector.collect_samples(duration=duration)
            
            if eeg_data:
                # Process the raw data into frequency bands
                band_powers = self.muse_collector.process_to_frequency_bands(eeg_data)
                
                # Calculate stress from the band powers
                stress_level = self.stress_analyzer.calculate_stress_from_bands(band_powers)
            else:
                print("Failed to collect EEG data")
                return None
        else:
            # Use simulated data
            eeg_data = self.muse_collector.get_sample_data(duration=duration)
            stress_level = self.stress_analyzer.calculate_stress_level(eeg_data)
        
        # Get stress category and trend
        category = self.stress_analyzer.get_stress_category()
        trend = self.stress_analyzer.get_recent_stress_trend()
        
        print(f"Stress Level: {stress_level:.1f}/100 ({category})")
        print(f"Trend: {trend}")
        
        return {
            'stress_level': stress_level,
            'category': category,
            'trend': trend,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_coffee_recommendations(self):
        """Get coffee recommendations based on measured stress"""
        if self.stress_analyzer.current_stress is None:
            print("No stress measurement available. Please measure stress first.")
            return None
        
        stress_level = self.stress_analyzer.current_stress
        current_hour = datetime.now().hour
        
        print(f"\n=== Coffee Recommendations Based on Stress Level: {stress_level:.1f} ===")
        
        # Get dose recommendation
        recommended_dose = self.coffee_recommender.get_dose_recommendation(
            stress_level, hour_of_day=current_hour)
        print(f"Recommended Coffee Dose: {recommended_dose:.1f}g")
        
        # Get blend recommendations
        blend_recommendations = self.coffee_recommender.get_blend_recommendations(
            stress_level, time_of_day=current_hour)
        
        # Get rest time recommendation for top blend
        recommended_rest = self.coffee_recommender.get_rest_recommendation(
            stress_level, blend_recommendations[0]['id'])
        print(f"Recommended Bean Rest Time: {recommended_rest} days")
        
        print("\nRecommended Coffee Blends:")
        for i, blend in enumerate(blend_recommendations, 1):
            print(f"{i}. {blend['name']} - {blend['caffeine_level']} caffeine")
            print(f"   Notes: {blend['notes']}")
            print(f"   Recommendation reason: {blend['reason']}")
        
        return {
            'stress_level': stress_level,
            'recommended_dose': recommended_dose,
            'recommended_rest_days': recommended_rest,
            'blend_recommendations': blend_recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def visualize_stress_history(self):
        """Visualize stress history"""
        if not self.stress_analyzer.stress_history:
            print("No stress history available")
            return
        
        # Extract data for plotting
        times, levels = zip(*self.stress_analyzer.stress_history)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(times, levels, 'o-', color='#2E86C1', linewidth=2)
        plt.title('Stress Level History', fontsize=16)
        plt.ylabel('Stress Level (0-100)', fontsize=12)
        plt.xlabel('Time', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add stress categories
        plt.axhspan(0, 30, alpha=0.2, color='green', label='Low Stress')
        plt.axhspan(30, 70, alpha=0.2, color='yellow', label='Moderate Stress')
        plt.axhspan(70, 100, alpha=0.2, color='red', label='High Stress')
        
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/stress_history_{timestamp}.png"
        plt.savefig(filename, dpi=300)
        print(f"Stress history chart saved as '{filename}'")
        plt.close()
    
    def save_recommendations(self, recommendations):
        """Save recommendations to a JSON file"""
        if not recommendations:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Save to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/coffee_recommendations_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=2)
            
        print(f"Recommendations saved to '{filename}'")
    
    def close(self):
        """Clean up and close the experiment"""
        if self.use_real_muse:
            self.muse_collector.disconnect()
        else:
            self.muse_collector.disconnect()


def run_experiment(args):
    """Run the stress-based coffee recommendation experiment"""
    print("\n========== Stress-Based Coffee Recommendation Experiment ==========")
    print("Using real Muse headband for EEG data: " + 
          ("Yes" if args.use_real_muse and MUSE_LSL_AVAILABLE else "No (Simulation)"))
    
    # Create and setup experiment
    experiment = MuseStressExperiment(use_real_muse=args.use_real_muse)
    
    if not experiment.setup():
        print("Failed to setup experiment. Exiting.")
        return
    
    try:
        # Measure baseline stress
        print("\nMeasuring baseline stress level...")
        baseline = experiment.measure_stress(duration=args.measurement_duration)
        
        if baseline:
            # Get coffee recommendations
            recommendations = experiment.get_coffee_recommendations()
            
            if recommendations:
                # Save recommendations
                experiment.save_recommendations(recommendations)
            
            # If multiple measurements requested
            if args.measurements > 1:
                print(f"\nWill take {args.measurements-1} additional measurements...")
                
                for i in range(args.measurements-1):
                    time.sleep(args.interval)
                    print(f"\nTaking measurement {i+2} of {args.measurements}...")
                    experiment.measure_stress(duration=args.measurement_duration)
            
            # Visualize stress history
            experiment.visualize_stress_history()
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    finally:
        # Clean up
        experiment.close()
        print("\nExperiment completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run a stress-based coffee recommendation experiment with Muse headband')
    
    parser.add_argument('--use-real-muse', action='store_true',
                        help='Use a real Muse headband (if available)')
    parser.add_argument('--measurements', type=int, default=1,
                        help='Number of stress measurements to take')
    parser.add_argument('--measurement-duration', type=int, default=10,
                        help='Duration of each stress measurement in seconds')
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval between measurements in seconds')
    
    args = parser.parse_args()
    run_experiment(args) 