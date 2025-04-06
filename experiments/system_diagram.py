#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon
import matplotlib.patheffects as PathEffects

def create_system_diagram():
    """Create a visual diagram of the stress-based coffee recommendation system"""
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define colors
    colors = {
        'bg': '#f5f5f5',
        'muse': '#6C8EBF',
        'raspberry': '#B85450',
        'coffee_machine': '#82B366',
        'tablet': '#9673A6',
        'arrow': '#333333',
        'text': '#333333',
        'highlight': '#FFD966',
        'data_flow': '#D79B00'
    }
    
    # Set background color
    fig.patch.set_facecolor(colors['bg'])
    
    # Add title
    title = ax.text(50, 95, 'Stress-Based Coffee Recommendation System', 
                    fontsize=22, fontweight='bold', ha='center', va='center',
                    color=colors['text'])
    title.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])
    
    # Draw Muse 2 headband
    muse_x, muse_y = 15, 75
    draw_muse_headband(ax, muse_x, muse_y, colors['muse'])
    ax.text(muse_x, muse_y-10, 'Muse 2 EEG Headband', 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color=colors['text'])
    ax.text(muse_x, muse_y-14, 'Collects brainwave data\nto assess stress levels', 
            fontsize=9, ha='center', va='center', color=colors['text'])
    
    # Draw Raspberry Pi
    rpi_x, rpi_y = 50, 50
    draw_raspberry_pi(ax, rpi_x, rpi_y, colors['raspberry'])
    ax.text(rpi_x, rpi_y-12, 'Raspberry Pi', 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color=colors['text'])
    ax.text(rpi_x, rpi_y-16, 'Central processing & communication hub', 
            fontsize=9, ha='center', va='center', color=colors['text'])
    
    # Draw Coffee Machine
    coffee_x, coffee_y = 85, 30
    draw_coffee_machine(ax, coffee_x, coffee_y, colors['coffee_machine'])
    ax.text(coffee_x, coffee_y-10, 'Smart Coffee Machine', 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color=colors['text'])
    ax.text(coffee_x, coffee_y-14, 'Receives brewing instructions via\nWiFi/Bluetooth from Raspberry Pi', 
            fontsize=9, ha='center', va='center', color=colors['text'])
    
    # Draw Tablet/Display
    tablet_x, tablet_y = 85, 75
    draw_tablet(ax, tablet_x, tablet_y, colors['tablet'])
    ax.text(tablet_x, tablet_y-10, 'User Interface Display', 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color=colors['text'])
    ax.text(tablet_x, tablet_y-14, 'Shows recommendations and\ncontrols the brewing process', 
            fontsize=9, ha='center', va='center', color=colors['text'])
    
    # Draw arrows to show data flow
    # Muse to Raspberry Pi
    draw_arrow(ax, muse_x+8, muse_y-2, rpi_x-8, rpi_y+5, colors['data_flow'], 'Stress Data')
    
    # Raspberry Pi to Tablet
    draw_arrow(ax, rpi_x+10, rpi_y+5, tablet_x-10, tablet_y, colors['data_flow'], 'Coffee Recommendations')
    
    # Raspberry Pi to Coffee Machine
    draw_arrow(ax, rpi_x+7, rpi_y-5, coffee_x-10, coffee_y, colors['data_flow'], 'Brewing Commands')
    
    # Recommendation box with the 3 key outputs
    draw_recommendation_box(ax, 45, 75, colors['highlight'])
    
    # Add WiFi/Bluetooth indicators
    draw_wifi_symbol(ax, 67, 45, colors['data_flow'])
    ax.text(67, 41, 'WiFi/Bluetooth\nCommunication', 
            fontsize=8, ha='center', va='center', color=colors['text'])
    
    # Add system description
    description = (
        "The system collects brainwave data from the Muse 2 headband to assess stress levels. "
        "Based on this data, the system provides personalized coffee recommendations: optimal dose, "
        "bean resting time, and blend suggestions. These recommendations are displayed on the user interface "
        "and can be sent directly to a compatible smart coffee machine for brewing."
    )
    
    fig.text(0.5, 0.04, description, fontsize=10, ha='center', va='center', 
             color=colors['text'], wrap=True)
    
    # Save the diagram
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('coffee_system_diagram.png', dpi=300, bbox_inches='tight')
    print("System diagram saved as 'coffee_system_diagram.png'")
    plt.close()


def draw_muse_headband(ax, x, y, color):
    """Draw the Muse 2 headband"""
    # Headband arc
    theta = np.linspace(-30, 210, 100)
    r = 6
    headband_x = x + r * np.cos(np.radians(theta))
    headband_y = y + r * np.sin(np.radians(theta))
    ax.plot(headband_x, headband_y, color=color, linewidth=3)
    
    # Sensors
    sensor_positions = [(-30, 0), (0, 6), (30, 0)]
    for sx, sy in sensor_positions:
        sensor = Circle((x + sx, y + sy), 1.2, color=color, alpha=0.7)
        ax.add_patch(sensor)
    
    # Brain waves indication
    wave_x = np.linspace(x-5, x+5, 50)
    wave_y = y + 3 + 1.5 * np.sin(wave_x * 1.5)
    ax.plot(wave_x, wave_y, color=color, linewidth=1, alpha=0.7)


def draw_raspberry_pi(ax, x, y, color):
    """Draw a simplified Raspberry Pi"""
    # Board
    board = Rectangle((x-8, y-8), 16, 16, facecolor=color, edgecolor='black', alpha=0.8)
    ax.add_patch(board)
    
    # CPU
    cpu = Rectangle((x-3, y-3), 6, 6, facecolor='black', alpha=0.7)
    ax.add_patch(cpu)
    
    # Ports
    for i in range(4):
        port = Rectangle((x-7+i*4, y+6), 3, 1.5, facecolor='#D3D3D3', edgecolor='black', alpha=0.9)
        ax.add_patch(port)
    
    # LEDs
    for i in range(2):
        led = Circle((x+5, y-6+i*3), 0.5, color='#ffcc00', alpha=0.9)
        ax.add_patch(led)


def draw_coffee_machine(ax, x, y, color):
    """Draw a simplified coffee machine"""
    # Main body
    body = Rectangle((x-10, y-8), 20, 16, facecolor=color, edgecolor='black', alpha=0.8)
    ax.add_patch(body)
    
    # Top
    top = Rectangle((x-8, y+8), 16, 2, facecolor=color, edgecolor='black', alpha=0.8)
    ax.add_patch(top)
    
    # Spout
    spout_x = [x-2, x+2, x+1, x-1]
    spout_y = [y-8, y-8, y-12, y-12]
    spout = Polygon(np.column_stack([spout_x, spout_y]), closed=True, 
                    facecolor=color, edgecolor='black', alpha=0.8)
    ax.add_patch(spout)
    
    # Control panel
    panel = Rectangle((x-5, y), 10, 5, facecolor='#D3D3D3', edgecolor='black', alpha=0.9)
    ax.add_patch(panel)
    
    # Cup
    cup_x = [x-3, x+3, x+4, x-4]
    cup_y = [y-18, y-18, y-13, y-13]
    cup = Polygon(np.column_stack([cup_x, cup_y]), closed=True, 
                  facecolor='#D3D3D3', edgecolor='black', alpha=0.9)
    ax.add_patch(cup)
    
    # Wifi/Bluetooth indicator
    wifi = Circle((x+8, y+5), 1, color='#3498db', alpha=0.9)
    ax.add_patch(wifi)


def draw_tablet(ax, x, y, color):
    """Draw a tablet with thin bezel"""
    # Tablet body
    body = Rectangle((x-12, y-8), 24, 18, facecolor='black', edgecolor='black', alpha=0.8)
    ax.add_patch(body)
    
    # Screen
    screen = Rectangle((x-11, y-7), 22, 16, facecolor=color, edgecolor='black', alpha=0.7)
    ax.add_patch(screen)
    
    # Draw simplified UI elements on screen
    # Stress level indicator
    stress_bar = Rectangle((x-9, y+5), 18, 2, facecolor='#ff6b6b', edgecolor='black', alpha=0.7)
    ax.add_patch(stress_bar)
    ax.text(x, y+6, 'Stress Level: 68/100', fontsize=7, ha='center', va='center', color='white')
    
    # Coffee recommendations
    ax.text(x, y+2, 'RECOMMENDATIONS', fontsize=7, fontweight='bold', ha='center', va='center', color='black')
    ax.text(x, y, '1. Dose: 18.5g', fontsize=6, ha='center', va='center', color='black')
    ax.text(x, y-2, '2. Rest Time: 10 days', fontsize=6, ha='center', va='center', color='black')
    ax.text(x, y-4, '3. Blend: Calm Mind', fontsize=6, ha='center', va='center', color='black')
    
    # Brew button
    brew_button = Rectangle((x-8, y-6), 16, 3, facecolor='#2ecc71', edgecolor='black', alpha=0.7)
    ax.add_patch(brew_button)
    ax.text(x, y-5, 'BREW NOW', fontsize=7, fontweight='bold', ha='center', va='center', color='white')


def draw_arrow(ax, x1, y1, x2, y2, color, label=None):
    """Draw an arrow with optional label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                            arrowstyle='-|>', color=color, 
                            linewidth=2, mutation_scale=15)
    ax.add_patch(arrow)
    
    if label:
        # Calculate middle point
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Small offset to not cover the arrow
        offset_x = (y2 - y1) * 0.1
        offset_y = (x1 - x2) * 0.1
        
        # Add label
        ax.text(mid_x + offset_x, mid_y + offset_y, label, 
                fontsize=8, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


def draw_recommendation_box(ax, x, y, color):
    """Draw a box highlighting the three key recommendations"""
    # Background box
    box = Rectangle((x-20, y-5), 40, 25, facecolor=color, edgecolor='black', alpha=0.4)
    ax.add_patch(box)
    
    # Title
    ax.text(x, y+17, 'KEY RECOMMENDATIONS', fontsize=12, fontweight='bold', ha='center', va='center', color='#333')
    
    # Recommendations
    recommendations = [
        ('1. COFFEE DOSE', 'Personalized amount based on current stress level'),
        ('2. BEAN REST TIME', 'Optimal resting period for coffee beans'),
        ('3. BLEND SUGGESTIONS', 'Coffee blends matched to stress profile')
    ]
    
    for i, (title, desc) in enumerate(recommendations):
        y_pos = y + 10 - i * 8
        ax.text(x, y_pos, title, fontsize=10, fontweight='bold', ha='center', va='center', color='#333')
        ax.text(x, y_pos - 3, desc, fontsize=8, ha='center', va='center', color='#333')


def draw_wifi_symbol(ax, x, y, color):
    """Draw a WiFi symbol"""
    for i in range(3):
        arc = np.linspace(-45, 225, 100)
        r = 3 + i * 1.5
        wifi_x = x + r * np.cos(np.radians(arc))
        wifi_y = y + r * np.sin(np.radians(arc))
        ax.plot(wifi_x, wifi_y, color=color, linewidth=1.5, alpha=0.7)
    
    # Center point
    center = Circle((x, y), 0.8, color=color, alpha=0.9)
    ax.add_patch(center)


if __name__ == "__main__":
    create_system_diagram() 