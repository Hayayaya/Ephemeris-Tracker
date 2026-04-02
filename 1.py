import matplotlib
matplotlib.use('TkAgg') 

from skyfield.api import load, wgs84
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os

# --- OFFLINE CONFIGURATION ---
OFFLINE_CITIES = {
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "tokyo": (35.6762, 139.6503),
    "chennai": (13.0827, 80.2707),
    "sydney": (-33.8688, 151.2093),
}

EPHEMERIS_FILE = 'de422.bsp'

# --- STRICT OFFLINE LOADING ---
# 1. Force timescale to use built-in files only (no downloading deltat.tdb or leap_seconds)
ts = load.timescale(builtin=True)

# 2. Check for ephemeris locally before loading to prevent Skyfield from attempting a download
if not os.path.exists(EPHEMERIS_FILE):
    print(f"CRITICAL ERROR: {EPHEMERIS_FILE} not found.")
    print("This app is in OFFLINE mode and cannot download data.")
    print("Please place de422.bsp in the script folder.")
    exit()

planets = load(EPHEMERIS_FILE)
earth_obj = planets['earth']
sun = planets['sun']

bodies = [
    'mercury barycenter', 'venus barycenter', 'earth barycenter', 
    'mars barycenter', 'jupiter barycenter', 'saturn barycenter',
    'uranus barycenter', 'neptune barycenter'
]

def get_offline_location(city_name):
    name = city_name.lower().strip()
    return OFFLINE_CITIES.get(name, (0.0, 0.0))

def run_app():
    while True:
        print("\n" + "="*30)
        print("   Solar System Tracker (STRICT OFFLINE)")
        print("="*30)
        print("(Type 'exit' to quit)")
        
        time_choice = input("Use real-time tracking (y/n)? ").strip().lower()
        if time_choice == 'exit': break
        
        d_input, t_input = None, None
        if time_choice != 'y':
            d_input = input("Date (YYYY-MM-DD): ").strip()
            if d_input == 'exit': break
            t_input = input("Time (HH:MM): ").strip()
            if t_input == 'exit': break
        
        p_input = input("Planet (e.g., Mars): ").strip().lower()
        if p_input == 'exit': break
        loc_input = input("City (from list): ").strip()
        if loc_input == 'exit': break
        tz_input = input("Timezone (e.g., UTC): ").strip()
        if tz_input == 'exit': break

        try:
            lat, lon = get_offline_location(loc_input)
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 10))
            
            def update(frame):
                ax.clear()
                target_name = p_input.capitalize()
                
                # Time handling
                try:
                    local_tz = pytz.timezone(tz_input)
                except pytz.UnknownTimeZoneError:
                    local_tz = pytz.UTC

                if time_choice != 'y':
                    dt = local_tz.localize(datetime.strptime(f"{d_input} {t_input}", "%Y-%m-%d %H:%M"))
                    t_current = ts.from_datetime(dt)
                    time_label = f"{d_input} {t_input}"
                    title_prefix = "Snapshot"
                else:
                    dt_now = datetime.now(local_tz)
                    t_current = ts.from_datetime(dt_now)
                    time_label = dt_now.strftime("%Y-%m-%d %H:%M:%S")
                    title_prefix = "LIVE"

                # Calculations
                observer = earth_obj + wgs84.latlon(lat, lon)
                target_key = p_input if p_input in ['sun', 'moon'] else f'{p_input} barycenter'
                
                try:
                    astrometric = observer.at(t_current).observe(planets[target_key])
                    alt, az, dist = astrometric.apparent().altaz()
                except KeyError:
                    ax.text(0, 0, f"Error: '{p_input}' not in file.", ha='center', color='red')
                    return

                # Orbital Plotting
                current_coords = {}
                for body in bodies:
                    pos = sun.at(t_current).observe(planets[body]).position.au
                    name = body.split()[0].capitalize()
                    current_coords[name] = (pos[0], pos[1])

                # Draw Sun
                ax.scatter(0, 0, color='yellow', s=300, edgecolors='orange', zorder=5)
                
                # Draw Planets and Orbits
                ex, ey = current_coords['Earth']
                for name, (px, py) in current_coords.items():
                    is_target = (name == target_name)
                    color = 'cyan' if name == 'Earth' else ('lime' if is_target else 'white')
                    ax.scatter(px, py, s=100 if is_target or name=='Earth' else 40, color=color, zorder=10)
                    ax.text(px + 0.1, py + 0.1, name, fontsize=9, color=color)
                    
                    # Orbit line
                    r = (px**2 + py**2)**0.5
                    ax.add_patch(plt.Circle((0,0), r, color='white', fill=False, alpha=0.1))

                # Visual distance line
                if target_name in current_coords and target_name != 'Earth':
                    tx, ty = current_coords[target_name]
                    ax.plot([ex, tx], [ey, ty], color='yellow', linestyle='--', alpha=0.5)
                    ax.text((ex+tx)/2, (ey+ty)/2, f"{dist.km:,.0f} km", color='yellow', 
                            fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

                # Scaling
                limit = max((current_coords[target_name][0]**2 + current_coords[target_name][1]**2)**0.5 * 1.2, 2) if target_name in current_coords else 15
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.set_aspect('equal')
                ax.set_xlabel("Distance (AU)")
                ax.set_ylabel("Distance (AU)")
                plt.title(f"{title_prefix}: {target_name}\n{time_label}\nAlt: {alt.degrees:.2f}°", color='white')

            if time_choice == 'y':
                ani = animation.FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
                plt.show()
            else:
                update(0)
                plt.show()

        except Exception as e:
            print(f"Error encountered: {e}")

if __name__ == "__main__":
    run_app()
