import matplotlib
matplotlib.use('TkAgg') 

from skyfield.api import load, wgs84
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geopy.geocoders import Nominatim
import ssl
import certifi
import time
import numpy as np

# Mass ratios (Planet / Sun) to calculate L2 distance
# Approx: r_l2 = R * (m / 3M)^(1/3)
MASS_RATIOS = {
    'Mercury': 1.66e-7, 'Venus': 2.44e-6, 'Earth': 3.00e-6,
    'Mars': 3.22e-7, 'Jupiter': 9.54e-4, 'Saturn': 2.85e-4,
    'Uranus': 4.36e-5, 'Neptune': 5.15e-5
}

try:
    ctx = ssl.create_default_context(cafile=certifi.where())
    geolocator = Nominatim(user_agent="planet_finder_v3", ssl_context=ctx)
except:
    geolocator = Nominatim(user_agent="planet_finder_v3")

ts = load.timescale()
planets = load('de422.bsp')
earth_obj = planets['earth']
sun = planets['sun']

bodies = [
    'mercury barycenter', 'venus barycenter', 'earth barycenter', 
    'mars barycenter', 'jupiter barycenter', 'saturn barycenter',
    'uranus barycenter', 'neptune barycenter'
]

def run_app():
    while True:
        print("\n" + "="*30)
        print("   Solar System Tracker & L2 Zoom")
        print("="*30)
        print("(Type 'exit' at any prompt to quit)")
        
        time_choice = input("Use real-time tracking (Today/Now)? (y/n): ").strip().lower()
        if time_choice == 'exit': break
        
        if time_choice != 'y':
            d_input = input("Enter Date (YYYY-MM-DD): ").strip()
            if d_input == 'exit': break
            t_input = input("Enter Time (HH:MM): ").strip()
            if t_input == 'exit': break
        
        p_input = input("Enter Target Planet (e.g., Mars, Jupiter): ").strip().lower()
        if p_input == 'exit': break
        loc_input = input("Enter City: ").strip()
        if loc_input == 'exit': break
        tz_input = input("Enter Timezone (e.g., UTC): ").strip()
        if tz_input == 'exit': break

        try:
            location = geolocator.geocode(loc_input, timeout=10)
            if not location:
                print("Error: Location not found.")
                continue 

            plt.style.use('dark_background')
            # Create two subplots side-by-side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            def update(frame):
                ax1.clear()
                ax2.clear()
                target_name = p_input.capitalize()
                
                # --- Time Handling ---
                if time_choice != 'y':
                    local_tz = pytz.timezone(tz_input)
                    dt = local_tz.localize(datetime.strptime(f"{d_input} {t_input}", "%Y-%m-%d %H:%M"))
                    t_current = ts.from_datetime(dt)
                    time_label = f"{d_input} {t_input}"
                    title_prefix = "Prediction"
                else:
                    dt_now = datetime.now(pytz.timezone(tz_input))
                    t_current = ts.from_datetime(dt_now)
                    time_label = dt_now.strftime("%Y-%m-%d %H:%M:%S")
                    title_prefix = "LIVE"

                # --- Data Fetching ---
                observer = earth_obj + wgs84.latlon(location.latitude, location.longitude)
                target_key = p_input if p_input in ['sun', 'moon'] else f'{p_input} barycenter'
                astrometric = observer.at(t_current).observe(planets[target_key])
                alt, az, dist = astrometric.apparent().altaz()
                
                current_coords = {}
                for body in bodies:
                    pos = sun.at(t_current).observe(planets[body]).position.au
                    name = body.split()[0].capitalize()
                    current_coords[name] = np.array([pos[0], pos[1]])

                ex, ey = current_coords['Earth']
                
                # --- Plot 1: Solar System View ---
                ax1.scatter(0, 0, color='yellow', s=200, edgecolors='orange', zorder=5) # Sun
                for name, pos in current_coords.items():
                    is_target = (name == target_name)
                    color = 'cyan' if name == 'Earth' else ('lime' if is_target else 'white')
                    ax1.scatter(pos[0], pos[1], s=80 if is_target or name=='Earth' else 30, color=color, zorder=10)
                    ax1.text(pos[0] + 0.1, pos[1] + 0.1, name, fontsize=8, color=color)
                    r = np.linalg.norm(pos)
                    ax1.add_patch(plt.Circle((0,0), r, color='white', fill=False, alpha=0.1))

                limit = max(np.linalg.norm(current_coords[target_name]) * 1.2, 2) if target_name in current_coords else 15
                ax1.set_xlim(-limit, limit)
                ax1.set_ylim(-limit, limit)
                ax1.set_aspect('equal')
                ax1.set_title(f"{title_prefix}: Solar System\n{time_label}", fontsize=10)

                # --- L2 Calculation & Plot 2: Zoom View ---
                if target_name in current_coords and target_name in MASS_RATIOS:
                    # Planet vector from Sun
                    P_vec = current_coords[target_name]
                    R_dist = np.linalg.norm(P_vec)
                    
                    # L2 distance from planet
                    l2_dist_au = R_dist * (MASS_RATIOS[target_name] / 3)**(1/3)
                    
                    # L2 is further out from the sun than the planet along the same vector
                    unit_vec = P_vec / R_dist
                    l2_vec = P_vec + (unit_vec * l2_dist_au)
                    
                    # Plot Earth, Planet, and L2 in Zoom
                    ax2.scatter(ex, ey, s=200, color='cyan', label='Earth')
                    ax2.scatter(P_vec[0], P_vec[1], s=150, color='lime', label=target_name)
                    ax2.scatter(l2_vec[0], l2_vec[1], s=100, color='red', marker='x', label=f'{target_name} L2')
                    
                    # Draw Line from Earth to L2
                    ax2.plot([ex, l2_vec[0]], [ey, l2_vec[1]], color='yellow', linestyle='--', alpha=0.5)
                    
                    # Setting zoom limits around the target and Earth
                    zoom_center = (P_vec + np.array([ex, ey])) / 2
                    zoom_range = max(np.linalg.norm(P_vec - np.array([ex, ey])), l2_dist_au * 5) * 0.8
                    
                    ax2.set_xlim(zoom_center[0] - zoom_range, zoom_center[1] + zoom_range) # Approximate window
                    ax2.set_xlim(min(ex, P_vec[0], l2_vec[0]) - 0.5, max(ex, P_vec[0], l2_vec[0]) + 0.5)
                    ax2.set_ylim(min(ey, P_vec[1], l2_vec[1]) - 0.5, max(ey, P_vec[1], l2_vec[1]) + 0.5)
                    
                    ax2.legend(loc='upper right', fontsize=8)
                    ax2.set_title(f"Zoom: Earth to {target_name} L2 Point\nAlt: {alt.degrees:.2f}°", fontsize=10)
                    ax2.set_aspect('equal')
                else:
                    ax2.text(0.5, 0.5, "L2 Zoom not available for this body", ha='center')

            if time_choice == 'y':
                ani = animation.FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
                plt.tight_layout()
                plt.show()
            else:
                update(0)
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error: {e}")

    print("\nExiting Tracker.")

if __name__ == "__main__":
    run_app()