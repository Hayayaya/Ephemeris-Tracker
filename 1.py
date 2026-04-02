import matplotlib
matplotlib.use('TkAgg') 

from skyfield.api import load, wgs84
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

# --- OFFLINE LIBRARIES ---
import geonamescache
from timezonefinder import TimezoneFinder

# Initialize offline databases
gc = geonamescache.GeonamesCache()
tf = TimezoneFinder()
ALL_CITIES = gc.get_cities()

# --- OFFLINE CONFIGURATION ---
EPHEMERIS_FILE = 'de422.bsp'
if not os.path.exists(EPHEMERIS_FILE):
    print(f"CRITICAL ERROR: {EPHEMERIS_FILE} not found.")
    exit()

ts = load.timescale(builtin=True)
planets = load(EPHEMERIS_FILE)
earth_obj = planets['earth']
sun = planets['sun']

MASS_RATIOS = {
    'mercury': 1.66e-7, 'venus': 2.44e-6, 'earth': 3.00e-6,
    'mars': 3.22e-7, 'jupiter': 9.54e-4, 'saturn': 2.85e-4,
    'uranus': 4.36e-5, 'neptune': 5.15e-5
}

bodies = [
    'mercury barycenter', 'venus barycenter', 'earth barycenter', 
    'mars barycenter', 'jupiter barycenter', 'saturn barycenter',
    'uranus barycenter', 'neptune barycenter'
]

def get_offline_location_data(city_query):
    search_name = city_query.title().strip()
    matches = [c for c in ALL_CITIES.values() if c['name'] == search_name]
    if not matches:
        return 51.4769, 0.0005, "UTC"
    target = max(matches, key=lambda x: x['population'])
    lat, lon = float(target['latitude']), float(target['longitude'])
    tz_str = tf.timezone_at(lng=lon, lat=lat) or "UTC"
    return lat, lon, tz_str

def get_angle(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))

def run_app():
    while True:
        print("\n" + "="*40)
        print("   SOLAR TRACKER: DISTANCE & L2 ANGLES")
        print("="*40)
        
        time_choice = input("Use real-time tracking (y/n)? ").strip().lower()
        if time_choice == 'exit': break
        
        d_input, t_input = None, None
        if time_choice != 'y':
            d_input = input("Date (YYYY-MM-DD): ").strip()
            t_input = input("Time (HH:MM): ").strip()
        
        p_input = input("Target Planet (e.g. Mars): ").strip().lower()
        loc_input = input("City Name: ").strip()

        try:
            lat, lon, tz_str = get_offline_location_data(loc_input)
            local_tz = pytz.timezone(tz_str)

            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), num="Planet & L2 Tracker")
            
            plt.subplots_adjust(bottom=0.15, top=0.85, wspace=0.3)
            
            def update(frame):
                ax1.clear()
                ax2.clear()
                target_name = p_input.capitalize()
                
                if time_choice == 'y':
                    dt_now = datetime.now(local_tz)
                    t_current = ts.from_datetime(dt_now)
                    time_label = dt_now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    ax1.text(0.02, 0.96, 'LIVE', transform=ax1.transAxes, color='red', 
                             fontweight='bold', fontsize=12, verticalalignment='top')
                else:
                    dt = local_tz.localize(datetime.strptime(f"{d_input} {t_input}", "%Y-%m-%d %H:%M"))
                    t_current = ts.from_datetime(dt)
                    time_label = f"{d_input} {t_input}"

                # Coordinates
                current_coords = {}
                for body in bodies:
                    pos = sun.at(t_current).observe(planets[body]).position.au
                    name = body.split()[0].capitalize()
                    current_coords[name] = np.array([pos[0], pos[1]])

                e_vec = current_coords['Earth']
                p_vec = current_coords.get(target_name, np.array([0,0]))
                
                dist_km = np.linalg.norm(p_vec - e_vec) * 149597870.7
                angle_e_p = get_angle(e_vec, p_vec)

                # --- Plot 1: Solar System ---
                ax1.scatter(0, 0, color='yellow', s=200, edgecolors='orange')
                for name, pos in current_coords.items():
                    is_active = name in ['Earth', target_name]
                    ax1.scatter(pos[0], pos[1], s=60 if is_active else 30, 
                                color='cyan' if name == 'Earth' else ('lime' if name == target_name else 'gray'),
                                alpha=1.0 if is_active else 0.3)
                    ax1.text(pos[0]+0.1, pos[1]+0.1, name, fontsize=8)

                ax1.plot([e_vec[0], p_vec[0]], [e_vec[1], p_vec[1]], 'w--', alpha=0.3)
                
                ax1.text(0.5, -0.12, f"Earth to {target_name}: {dist_km:,.0f} km", 
                         transform=ax1.transAxes, ha='center', color='yellow', fontsize=11, fontweight='bold')
                
                limit = max(np.linalg.norm(p_vec) * 1.2, 2.5)
                ax1.set_xlim(-limit, limit); ax1.set_ylim(-limit, limit)
                ax1.set_aspect('equal')
                ax1.set_title(f"Solar System\n{time_label}", pad=20)

                if p_input in MASS_RATIOS:
                    R_dist = np.linalg.norm(p_vec)
                    l2_au = R_dist * (MASS_RATIOS[p_input] / 3)**(1/3)
                    l2_km = l2_au * 149597870.7
                    l2_vec = p_vec + (p_vec / R_dist * l2_au)
                    angle_e_l2 = get_angle(e_vec, l2_vec)
                    
                    ax2.scatter(p_vec[0], p_vec[1], s=400, color='lime', label=target_name)
                    ax2.scatter(l2_vec[0], l2_vec[1], s=200, color='red', marker='x', label='L2 Point')
                    ax2.plot([p_vec[0], l2_vec[0]], [p_vec[1], l2_vec[1]], 'w:', alpha=0.5)
                    
                    ax2.text((p_vec[0]+l2_vec[0])/2, (p_vec[1]+l2_vec[1])/2, f"{l2_km:,.0f} km", 
                             color='tomato', fontsize=9, ha='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, lw=0))
                    
                    margin = l2_au * 1.5
                    ax2.set_xlim(min(p_vec[0], l2_vec[0]) - margin, max(p_vec[0], l2_vec[0]) + margin)
                    ax2.set_ylim(min(p_vec[1], l2_vec[1]) - margin, max(p_vec[1], l2_vec[1]) + margin)
                    ax2.set_aspect('equal')
                    ax2.set_title(f"L2 Zoom: {target_name}", pad=20)
                    ax2.text(0.5, -0.12, f"Angle E-S-P: {angle_e_p:.2f}°\nAngle E-S-L2: {angle_e_l2:.2f}°", 
                             transform=ax2.transAxes, ha='center', color='white', fontsize=11)
                    ax2.legend(loc='upper right')
                else:
                    ax2.text(0.5, 0.5, "L2 data not available", ha='center')

            if time_choice == 'y':
                ani = animation.FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
                plt.show()
            else:
                update(0)
                plt.show()

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_app()
