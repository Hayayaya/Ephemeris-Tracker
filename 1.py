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

# Dictionary of planet masses relative to Sun for L2 calculation
# Formula: r_L2 = R * (M_planet / (3 * M_sun))^(1/3)
MASS_RATIOS = {
    'Mercury': 1.6601e-7, 'Venus': 2.4478e-6, 'Earth': 3.0034e-6,
    'Mars': 3.2271e-7, 'Jupiter': 9.5479e-4, 'Saturn': 2.8588e-4,
    'Uranus': 4.3662e-5, 'Neptune': 5.1513e-5
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
        print("    Solar System Tracker + L2")
        print("="*30)
        print("(Type 'exit' at any prompt to quit)")
        
        time_choice = input("Use real-time tracking (Today/Now)? (y/n): ").strip().lower()
        if time_choice == 'exit': break
        
        if time_choice != 'y':
            d_input = input("Enter Date (YYYY-MM-DD): ").strip()
            if d_input == 'exit': break
            t_input = input("Enter Time (HH:MM): ").strip()
            if t_input == 'exit': break
        
        p_input = input("Enter Planet (e.g., Mars, Neptune): ").strip().lower()
        if p_input == 'exit': break
        loc_input = input("Enter City: ").strip()
        if loc_input == 'exit': break
        tz_input = input("Enter Timezone (e.g., UTC): ").strip()
        if tz_input == 'exit': break

        try:
            location = None
            for attempt in range(3):
                try:
                    location = geolocator.geocode(loc_input, timeout=10)
                    if location: break
                except:
                    time.sleep(1)
            if not location:
                print("Error: Location not found.")
                continue 

            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 10))
            
            def update(frame):
                ax.clear()
                target_name = p_input.capitalize()
                
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

                observer = earth_obj + wgs84.latlon(location.latitude, location.longitude)
                target_key = p_input if p_input in ['sun', 'moon'] else f'{p_input} barycenter'
                astrometric = observer.at(t_current).observe(planets[target_key])
                alt, az, dist = astrometric.apparent().altaz()
                
                current_coords = {}
                for body in bodies:
                    pos = sun.at(t_current).observe(planets[body]).position.au
                    name = body.split()[0].capitalize()
                    current_coords[name] = np.array([pos[0], pos[1]])

                # Plot Sun
                ax.scatter(0, 0, color='yellow', s=300, edgecolors='orange', zorder=5)
                
                ex, ey = current_coords['Earth']
                
                for name, pos in current_coords.items():
                    px, py = pos
                    is_target = (name == target_name)
                    color = 'cyan' if name == 'Earth' else ('lime' if is_target else 'white')
                    ax.scatter(px, py, s=100 if is_target or name=='Earth' else 40, color=color, zorder=10)
                    ax.text(px + 0.1, py + 0.1, name, fontsize=9, color=color)
                    
                    # Draw Orbits
                    r = np.linalg.norm(pos)
                    ax.add_patch(plt.Circle((0,0), r, color='white', fill=False, alpha=0.1))

                    # --- L2 LAGRANGE CALCULATION ---
                    if is_target and name in MASS_RATIOS:
                        # R = distance from sun to planet
                        R = np.linalg.norm(pos)
                        # r_l2 = distance from planet to L2
                        r_l2 = R * (MASS_RATIOS[name] / 3)**(1/3)
                        
                        # L2 is on the line connecting Sun (0,0) and Planet, but further out.
                        # Unit vector from Sun to Planet
                        unit_vec = pos / R
                        l2_pos = pos + (unit_vec * r_l2)
                        
                        ax.scatter(l2_pos[0], l2_pos[1], color='red', s=30, marker='x', zorder=15)
                        ax.text(l2_pos[0]+0.05, l2_pos[1]+0.05, f"{name} L2", color='red', fontsize=8)
                        
                        l2_dist_km = r_l2 * 149597870.7 # AU to KM
                        print(f"Calculated {name} L2 distance from planet: {l2_dist_km:,.0f} km")

                # Distance line from Earth to Target
                if target_name in current_coords and target_name != 'Earth':
                    tx, ty = current_coords[target_name]
                    ax.plot([ex, tx], [ey, ty], color='yellow', linestyle='--', alpha=0.5)
                    ax.text((ex+tx)/2, (ey+ty)/2, f"{dist.km:,.0f} km", color='yellow', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

                limit = max(np.linalg.norm(current_coords[target_name]) * 1.3, 2) if target_name in current_coords else 15
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.set_aspect('equal')
                plt.title(f"{title_prefix}: Earth to {target_name}\n{time_label}\nAlt: {alt.degrees:.2f}°", color='white')

            if time_choice == 'y':
                ani = animation.FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
                plt.show()
            else:
                update(0)
                plt.show()

        except Exception as e:
            print(f"Error: {e}")

    print("\nExiting Tracker.")

if __name__ == "__main__":
    run_app()