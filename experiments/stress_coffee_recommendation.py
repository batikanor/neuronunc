#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import random
import threading
import socketserver
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

"""
Proof of concept for a stress-based coffee recommendation system
- Reads EEG data from Muse 2 headband
- Analyzes stress levels
- Recommends coffee parameters (dose, resting time, blends)
- Interfaces with coffee machine via Raspberry Pi
"""

# Simulated Muse 2 data collection
class MuseDataCollector:
    """Simulates collecting data from Muse 2 headband"""
    
    def __init__(self):
        self.connected = False
        self.data_buffer = []
        self.is_recording = False
        self.channels = ['TP9', 'AF7', 'AF8', 'TP10', 'AUX']
        self.frequencies = {'delta': (1, 4), 'theta': (4, 8), 
                           'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
    
    def connect(self):
        """Simulate connecting to Muse 2 headband"""
        print("Connecting to Muse 2 headband...")
        time.sleep(1.5)  # Simulate connection delay
        self.connected = True
        print("Connected to Muse 2")
        return self.connected
    
    def disconnect(self):
        """Disconnect from headband"""
        if self.connected:
            print("Disconnecting from Muse 2...")
            time.sleep(0.5)
            self.connected = False
            self.is_recording = False
            print("Disconnected")
    
    def start_recording(self):
        """Start collecting EEG data"""
        if not self.connected:
            print("Error: Not connected to headband")
            return False
        
        self.is_recording = True
        self.data_buffer = []
        print("Started recording EEG data")
        return True
    
    def stop_recording(self):
        """Stop recording EEG data"""
        if self.is_recording:
            self.is_recording = False
            print("Stopped recording")
            return True
        return False
    
    def get_sample_data(self, duration=5):
        """Get simulated EEG data for a specified duration"""
        if not self.connected:
            print("Error: Not connected to headband")
            return None
        
        print(f"Collecting {duration} seconds of EEG data...")
        
        # Simulate collecting data for the specified duration
        eeg_data = {}
        timestamps = np.arange(0, duration, 1/256)  # 256 Hz sampling rate
        
        # Generate simulated data for each frequency band
        for band, (low_freq, high_freq) in self.frequencies.items():
            # Add some randomness to simulate different stress levels
            amplitude = np.random.uniform(0.5, 2.0)
            if band == 'alpha':  # Alpha waves are inversely related to stress
                amplitude = np.random.uniform(0.3, 1.8)  # Lower alpha = higher stress
            elif band == 'beta':  # Beta waves increase with stress
                amplitude = np.random.uniform(0.8, 2.5)
            
            # Generate band data with some noise
            eeg_data[band] = amplitude * np.sin(2 * np.pi * low_freq * timestamps) + \
                            0.5 * amplitude * np.sin(2 * np.pi * high_freq * timestamps) + \
                            0.2 * np.random.randn(len(timestamps))
        
        print("Data collection complete")
        return eeg_data


class StressAnalyzer:
    """Analyzes EEG data to determine stress levels"""
    
    def __init__(self):
        self.stress_history = []
        self.current_stress = None
    
    def calculate_stress_level(self, eeg_data):
        """Calculate stress level from EEG data
        
        Real implementation would use proper EEG analysis algorithms.
        This is a simplified approximation based on alpha/beta ratio
        which is commonly used in stress research.
        """
        if not eeg_data:
            return None
        
        # Calculate mean power in each frequency band
        alpha_power = np.mean(np.power(eeg_data['alpha'], 2))
        beta_power = np.mean(np.power(eeg_data['beta'], 2))
        theta_power = np.mean(np.power(eeg_data['theta'], 2))
        
        # Alpha/Beta ratio - lower values typically indicate higher stress
        # Higher beta and lower alpha = more stress
        alpha_beta_ratio = alpha_power / (beta_power + 1e-10)  # Avoid division by zero
        
        # Normalize to a 0-100 stress scale (0 = relaxed, 100 = highly stressed)
        # This is a simplified model - real implementation would be trained on actual EEG data
        stress_level = 100 * (1 - (alpha_beta_ratio / 5))
        stress_level = max(0, min(100, stress_level))  # Clamp between 0-100
        
        # Additional factor: theta waves increase during drowsiness/meditation
        if theta_power > 1.5 * alpha_power:
            stress_level *= 0.7  # Reduce stress estimate if showing meditation signs
        
        self.current_stress = stress_level
        self.stress_history.append((datetime.now(), stress_level))
        
        return stress_level
    
    def get_stress_category(self):
        """Convert numerical stress level to category"""
        if self.current_stress is None:
            return "Unknown"
        
        if self.current_stress < 30:
            return "Low"
        elif self.current_stress < 70:
            return "Moderate"
        else:
            return "High"
    
    def get_recent_stress_trend(self, days=7):
        """Analyze recent stress trend (simulated for demo)"""
        # In a real app, we'd use actual historical data
        # For demo, generate some random historical data
        if not self.stress_history:
            return "No stress history available"
        
        # Use the last value as current and simulate some history
        current = self.current_stress
        
        # Simple trend: comparing current to average of recent history
        if len(self.stress_history) > 1:
            recent_avg = np.mean([s for _, s in self.stress_history[:-1]])
            if current < recent_avg - 10:
                return "Decreasing (Improving)"
            elif current > recent_avg + 10:
                return "Increasing (Worsening)"
            else:
                return "Stable"
        
        return "Insufficient data for trend analysis"


class CoffeeRecommender:
    """Recommends coffee parameters based on stress level"""
    
    def __init__(self):
        # Coffee database - in production this would come from a real database
        self.coffee_database = {
            'blends': [
                {'id': 1, 'name': 'Calm Mind', 'notes': 'Smooth, low acidity, chocolate notes', 
                 'caffeine': 'Low', 'acidity': 'Low', 'stress_effect': -5},
                {'id': 2, 'name': 'Balanced Day', 'notes': 'Medium body, nutty with caramel sweetness', 
                 'caffeine': 'Medium', 'acidity': 'Medium', 'stress_effect': 0},
                {'id': 3, 'name': 'Focus Blend', 'notes': 'Full-bodied, dark chocolate, hint of citrus', 
                 'caffeine': 'Medium-High', 'acidity': 'Medium', 'stress_effect': 5},
                {'id': 4, 'name': 'Energize', 'notes': 'Bright, fruity, high clarity', 
                 'caffeine': 'High', 'acidity': 'High', 'stress_effect': 10},
                {'id': 5, 'name': 'Soothe', 'notes': 'Mild body, subtle floral notes, honey sweetness', 
                 'caffeine': 'Low', 'acidity': 'Very Low', 'stress_effect': -10}
            ]
        }
        
        # Trained models (simplified for demo)
        # In production, these would be actual trained models
        self.dose_model = self._create_dummy_dose_model()
        self.rest_time_model = self._create_dummy_rest_model()
    
    def _create_dummy_dose_model(self):
        """Create a dummy model for dose prediction"""
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        # In real life, this would be trained on actual data
        # Here we just fit it to some dummy data to make it callable
        X = np.array([[s, h, c] for s in range(0, 101, 10) 
                              for h in range(6, 24, 2)
                              for c in range(1, 6)]) 
        # Stress, hour of day, coffee id -> dose
        y = 18 + (-0.05 * X[:, 0]) + (0.2 * X[:, 1]) + (0.5 * X[:, 2]) + np.random.randn(X.shape[0]) * 0.5
        model.fit(X, y)
        return model
    
    def _create_dummy_rest_model(self):
        """Create a dummy model for rest time prediction"""
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        # Dummy model
        X = np.array([[s, c] for s in range(0, 101, 5) for c in range(1, 6)]) 
        # Stress, coffee id -> rest days
        y = 7 + (0.05 * X[:, 0]) + (1.2 * X[:, 1]) + np.random.randn(X.shape[0]) * 0.8
        model.fit(X, y)
        return model
    
    def get_dose_recommendation(self, stress_level, hour_of_day=None, preferred_coffee_id=None):
        """Get recommended coffee dose based on stress level"""
        if hour_of_day is None:
            hour_of_day = datetime.now().hour
        
        if preferred_coffee_id is None:
            # Choose a default coffee blend based on stress
            if stress_level < 30:
                preferred_coffee_id = 2  # Balanced blend for low stress
            elif stress_level < 70:
                preferred_coffee_id = 3  # Focus blend for medium stress
            else:
                preferred_coffee_id = 1  # Calming blend for high stress
        
        # Use model to predict dose
        features = np.array([[stress_level, hour_of_day, preferred_coffee_id]])
        predicted_dose = self.dose_model.predict(features)[0]
        
        # Round to reasonable coffee dose
        rounded_dose = round(predicted_dose * 2) / 2  # Round to nearest 0.5g
        
        # Apply adjustments based on time of day
        if hour_of_day >= 16:  # After 4pm
            rounded_dose = max(rounded_dose - 1.5, 14)  # Reduce dose later in day
        
        # Ensure dose is in reasonable range
        return max(14, min(22, rounded_dose))
    
    def get_rest_recommendation(self, stress_level, coffee_id):
        """Get recommended bean resting time based on stress and coffee"""
        features = np.array([[stress_level, coffee_id]])
        predicted_rest = self.rest_time_model.predict(features)[0]
        
        # Round to whole days
        rounded_rest = round(predicted_rest)
        
        # Ensure rest time is in reasonable range
        return max(3, min(21, rounded_rest))
    
    def get_blend_recommendations(self, stress_level, time_of_day=None):
        """Recommend coffee blends based on stress level and time of day"""
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        # Logic for recommendations
        blends = self.coffee_database['blends']
        
        # Rank blends based on appropriateness for current stress
        def score_blend(blend):
            # Base score on how well the blend counters the current stress
            stress_counter_score = 0
            if stress_level > 70:  # High stress - prefer calming
                stress_counter_score = -blend['stress_effect']  # Reverse effect
            elif stress_level < 30:  # Low stress - match preference
                stress_counter_score = 5 - abs(blend['stress_effect'])  # Middle blends score higher
            else:  # Medium stress
                stress_counter_score = 10 - abs(blend['stress_effect'])
            
            # Time of day adjustments
            time_score = 0
            if time_of_day < 10:  # Morning
                # Higher caffeine scores better in morning
                time_score = 10 if blend['caffeine'] == 'High' else \
                           7 if blend['caffeine'] == 'Medium-High' else \
                           5 if blend['caffeine'] == 'Medium' else 3
            elif time_of_day >= 15:  # Afternoon/Evening
                # Lower caffeine scores better later
                time_score = 10 if blend['caffeine'] == 'Low' else \
                           7 if blend['caffeine'] == 'Medium' else \
                           3 if blend['caffeine'] == 'Medium-High' else 1
            else:  # Mid-day
                # Medium caffeine is best
                time_score = 10 if blend['caffeine'] == 'Medium' else \
                            7 if blend['caffeine'] == 'Medium-High' else 5
            
            return stress_counter_score + time_score
        
        # Score and sort blends
        scored_blends = [(blend, score_blend(blend)) for blend in blends]
        scored_blends.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 recommendations with reasoning
        recommendations = []
        for blend, score in scored_blends[:3]:
            reason = ""
            if stress_level > 70:
                reason = f"This {blend['caffeine']} caffeine blend can help manage high stress"
            elif stress_level < 30:
                reason = f"This blend complements your currently relaxed state"
            else:
                reason = f"A balanced choice for your moderate stress level"
                
            # Add time context
            if time_of_day < 10:
                reason += " and is suitable for morning consumption"
            elif time_of_day >= 15:
                reason += " without disrupting your sleep cycle"
            
            recommendations.append({
                'id': blend['id'],
                'name': blend['name'],
                'notes': blend['notes'],
                'caffeine_level': blend['caffeine'],
                'reason': reason
            })
        
        return recommendations


class CoffeeMachineInterface:
    """Interface to communicate with smart coffee machine via Raspberry Pi"""
    
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.connected = False
        self.machine_status = "Unknown"
        self.last_brew_parameters = None
    
    def connect(self):
        """Simulate connecting to coffee machine via network"""
        print(f"Connecting to coffee machine at {self.host}:{self.port}...")
        time.sleep(1)  # Simulate connection delay
        self.connected = True
        self.machine_status = "Ready"
        print("Connected to coffee machine")
        return True
    
    def disconnect(self):
        """Disconnect from coffee machine"""
        if self.connected:
            print("Disconnecting from coffee machine...")
            time.sleep(0.5)
            self.connected = False
            self.machine_status = "Disconnected"
            print("Disconnected")
    
    def get_machine_status(self):
        """Get coffee machine status"""
        if not self.connected:
            return "Not connected"
        
        # Simulate getting status from machine
        return self.machine_status
    
    def send_brew_command(self, dose, blend_id, grind_size=None, temperature=None, pressure=None):
        """Send brewing parameters to coffee machine"""
        if not self.connected:
            print("Error: Not connected to coffee machine")
            return False
        
        # Set defaults for missing parameters
        if grind_size is None:
            grind_size = 18  # Default medium grind
        if temperature is None:
            temperature = 93  # Default brew temp in Celsius
        if pressure is None:
            pressure = 9  # Default pressure in bars
        
        # Prepare command
        command = {
            'command': 'brew',
            'parameters': {
                'dose_g': dose,
                'blend_id': blend_id,
                'grind_size': grind_size,
                'temperature_c': temperature,
                'pressure_bar': pressure,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Simulate sending command to machine
        print(f"Sending brew command to machine: {json.dumps(command, indent=2)}")
        time.sleep(0.8)  # Simulate network delay
        
        # Simulate machine response
        self.machine_status = "Brewing"
        self.last_brew_parameters = command['parameters']
        print("Command received by machine, brewing started")
        
        # In real implementation, this would return actual success/failure from machine
        return True


class AppInterface:
    """Simulate the user interface for the stress-coffee recommendation system"""
    
    def __init__(self, muse_collector, stress_analyzer, coffee_recommender, machine_interface):
        self.muse = muse_collector
        self.stress = stress_analyzer
        self.recommender = coffee_recommender
        self.machine = machine_interface
        self.current_recommendations = None
    
    def measure_stress(self, duration=5):
        """Measure current stress levels"""
        # Connect to Muse if not already connected
        if not self.muse.connected:
            self.muse.connect()
        
        # Get EEG data
        self.muse.start_recording()
        eeg_data = self.muse.get_sample_data(duration)
        self.muse.stop_recording()
        
        # Analyze stress
        stress_level = self.stress.calculate_stress_level(eeg_data)
        stress_category = self.stress.get_stress_category()
        trend = self.stress.get_recent_stress_trend()
        
        result = {
            'stress_level': stress_level,
            'category': stress_category,
            'trend': trend,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n=== Stress Measurement Results ===")
        print(f"Stress Level: {stress_level:.1f}/100 ({stress_category})")
        print(f"Trend: {trend}")
        
        return result
    
    def get_recommendations(self):
        """Get coffee recommendations based on current stress level"""
        if self.stress.current_stress is None:
            print("Error: No stress measurement available. Please measure stress first.")
            return None
        
        stress_level = self.stress.current_stress
        current_hour = datetime.now().hour
        
        # Get dose recommendation
        recommended_dose = self.recommender.get_dose_recommendation(
            stress_level, hour_of_day=current_hour)
        
        # Get blend recommendations
        blend_recommendations = self.recommender.get_blend_recommendations(
            stress_level, time_of_day=current_hour)
        
        # Get rest time recommendation for top blend
        recommended_rest = self.recommender.get_rest_recommendation(
            stress_level, blend_recommendations[0]['id'])
        
        self.current_recommendations = {
            'stress_level': stress_level,
            'recommended_dose': recommended_dose,
            'recommended_rest_days': recommended_rest,
            'blend_recommendations': blend_recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n=== Coffee Recommendations ===")
        print(f"Based on stress level: {stress_level:.1f}/100 ({self.stress.get_stress_category()})")
        print(f"Recommended dose: {recommended_dose:.1f}g")
        print(f"Recommended bean rest time: {recommended_rest} days")
        print("\nRecommended coffee blends:")
        for i, blend in enumerate(blend_recommendations, 1):
            print(f"{i}. {blend['name']} - {blend['caffeine_level']} caffeine")
            print(f"   Notes: {blend['notes']}")
            print(f"   Recommendation reason: {blend['reason']}")
        
        return self.current_recommendations
    
    def brew_coffee(self, blend_choice=0):
        """Brew coffee with recommended parameters"""
        if self.current_recommendations is None:
            print("Error: No recommendations available. Please get recommendations first.")
            return False
        
        if not self.machine.connected:
            self.machine.connect()
        
        # Use the selected blend or default to the first recommendation
        if blend_choice < 0 or blend_choice >= len(self.current_recommendations['blend_recommendations']):
            blend_choice = 0
        
        selected_blend = self.current_recommendations['blend_recommendations'][blend_choice]
        dose = self.current_recommendations['recommended_dose']
        
        # Default grind size based on blend (just for demo)
        grind_size = 15 if selected_blend['caffeine_level'] == 'High' else \
                    18 if selected_blend['caffeine_level'] == 'Medium' else 20
        
        # Send brew command
        success = self.machine.send_brew_command(
            dose=dose,
            blend_id=selected_blend['id'],
            grind_size=grind_size
        )
        
        if success:
            print(f"\nBrewing {selected_blend['name']} with {dose:.1f}g dose")
            print(f"Grind size: {grind_size}")
            print(f"This coffee should help manage your {self.stress.get_stress_category().lower()} stress levels")
        
        return success
    
    def display_stress_history(self):
        """Display stress history chart"""
        if not self.stress.stress_history:
            print("No stress history available")
            return
        
        # Extract data for plotting
        times, levels = zip(*self.stress.stress_history)
        
        # Create plot
        plt.figure(figsize=(10, 5))
        plt.plot(times, levels, 'o-', color='#2E86C1')
        plt.title('Stress Level History')
        plt.ylabel('Stress Level (0-100)')
        plt.xlabel('Time')
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add stress categories
        plt.axhspan(0, 30, alpha=0.2, color='green', label='Low Stress')
        plt.axhspan(30, 70, alpha=0.2, color='yellow', label='Moderate Stress')
        plt.axhspan(70, 100, alpha=0.2, color='red', label='High Stress')
        
        plt.legend()
        plt.tight_layout()
        
        # In a real app, this would display in the UI
        # For this proof of concept, we'll just save the plot
        plt.savefig('stress_history.png')
        print("Stress history chart saved as 'stress_history.png'")
        plt.close()


def run_demo():
    """Run a demonstration of the system"""
    print("\n=== Stress-Based Coffee Recommendation System Demo ===\n")
    
    # Initialize components
    muse = MuseDataCollector()
    stress_analyzer = StressAnalyzer()
    coffee_recommender = CoffeeRecommender()
    machine_interface = CoffeeMachineInterface()
    
    app = AppInterface(muse, stress_analyzer, coffee_recommender, machine_interface)
    
    # Simulate a user session
    print("Starting user session...\n")
    
    # Measure stress
    stress_result = app.measure_stress(duration=3)
    time.sleep(1)
    
    # Get recommendations
    recommendations = app.get_recommendations()
    time.sleep(1)
    
    # Brew coffee with the top recommendation
    app.brew_coffee(blend_choice=0)
    
    # Simulate another measurement after some time
    print("\nSimulating another measurement after coffee consumption...\n")
    time.sleep(2)
    
    # Adjust the simulated stress level to be lower after coffee
    # In a real implementation, this would be an actual new measurement
    current = stress_analyzer.current_stress
    # Simulate stress reduction after coffee (in reality this would be a new measurement)
    new_stress = max(10, current * 0.7)  # Reduce stress by ~30%
    stress_analyzer.stress_history.append((datetime.now(), new_stress))
    stress_analyzer.current_stress = new_stress
    
    print(f"Follow-up Stress Level: {new_stress:.1f}/100 ({stress_analyzer.get_stress_category()})")
    print(f"Change: {current - new_stress:.1f} points improvement")
    
    # Display stress history
    app.display_stress_history()
    
    # Clean up
    muse.disconnect()
    machine_interface.disconnect()
    
    print("\nDemo completed!")


# Raspberry Pi Server Simulation (for communicating with coffee machine)
class CoffeeMachineHandler(socketserver.BaseRequestHandler):
    """Handler for coffee machine communication"""
    
    def handle(self):
        data = self.request.recv(1024).strip()
        command = pickle.loads(data)
        
        print(f"Received command from app: {command}")
        
        # Process command (in real implementation)
        response = {'status': 'success', 'message': 'Command executed'}
        
        self.request.sendall(pickle.dumps(response))


def start_raspberry_pi_server():
    """Simulate starting a server on Raspberry Pi to interface with coffee machine"""
    HOST, PORT = "localhost", 8000
    
    # Create the server
    server = socketserver.TCPServer((HOST, PORT), CoffeeMachineHandler)
    
    print(f"Starting Raspberry Pi server on {HOST}:{PORT}")
    print("Server will interface between app and coffee machine")
    print("Press Ctrl+C to stop")
    
    # In a real implementation, this would run in a separate thread
    # For demo purposes, we'll just show the server initialization
    # server.serve_forever()
    

if __name__ == "__main__":
    # Show Raspberry Pi server interface
    start_raspberry_pi_server()
    
    # Run the demo
    run_demo() 