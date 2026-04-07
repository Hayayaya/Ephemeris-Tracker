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

# --- CONSTANTS ---
EPHEMERIS_FILE = 'de422.bsp'
MU_SUN = 1.32712440018e11  # km^3/s^2
AU_KM = 149597870.7

if not os.path.exists(EPHEMERIS_FILE):
    print(f"CRITICAL ERROR: {EPHEMERIS_FILE} not found.")
    exit()

ts = load.timescale(builtin=True)
planets = load(EPHEMERIS_FILE)
earth_obj = planets['earth']
sun = planets['sun']

# Mass of planet / Mass of Sun
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

def calculate_hohmann(r1_au, r2_au):
    r1, r2 = r1_au * AU_KM, r2_au * AU_KM
    a_trans = (r1 + r2) / 2
    v_orbit1 = np.sqrt(MU_SUN / r1)
    v_orbit2 = np.sqrt(MU_SUN / r2)
    v_peri = np.sqrt(MU_SUN * (2/r1 - 1/a_trans))
    v_apo = np.sqrt(MU_SUN * (2/r2 - 1/a_trans))
    return abs(v_peri - v_orbit1) + abs(v_orbit2 - v_apo), a_trans / AU_KM

def run_app():
    while True:
        print("\n" + "="*40)
        print("    MISSION PLANNER: HOHMANN & L2 DISTANCE")
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
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), num="L2 & Transfer Visualization")
            plt.subplots_adjust(bottom=0.2)
            
            def update(frame):
                ax1.clear()
                ax2.clear()
                target_name = p_input.capitalize()
                
                if time_choice == 'y':
                    dt_now = datetime.now(local_tz)
                    t_current = ts.from_datetime(dt_now)
                    time_label = dt_now.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    dt = local_tz.localize(datetime.strptime(f"{d_input} {t_input}", "%Y-%m-%d %H:%M"))
                    t_current = ts.from_datetime(dt)
                    time_label = f"{d_input} {t_input}"

                # Planet Positions
                current_coords = {}
                for body in bodies:
                    pos = sun.at(t_current).observe(planets[body]).position.au
                    name = body.split()[0].capitalize()
                    current_coords[name] = np.array([pos[0], pos[1]])

                e_vec = current_coords['Earth']
                p_vec = current_coords.get(target_name, np.array([0,0]))
                r_e, r_p = np.linalg.norm(e_vec), np.linalg.norm(p_vec)
                
                # --- Plot 1: Solar System & Hohmann ---
                dv, a_trans = calculate_hohmann(r_e, r_p)
                ax1.scatter(0, 0, color='yellow', s=150, edgecolors='orange', label="Sun")
                
                for name, pos in current_coords.items():
                    # Set color/style based on whether the planet is Earth or the Target
                    p_color = 'cyan' if name == 'Earth' else ('lime' if name == target_name else 'gray')
                    p_alpha = 1.0 if name in ['Earth', target_name] else 0.5
                    
                    ax1.scatter(pos[0], pos[1], s=50, color=p_color, alpha=p_alpha)
                    
                    # ADDED: Planet Names with background boxes for visibility
                    ax1.annotate(name, (pos[0], pos[1]), 
                                 xytext=(5, 5), textcoords='offset points',
                                 color=p_color, fontsize=9, alpha=p_alpha,
                                 bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1))
                
                # Drawing Transfer Arc
                theta = np.linspace(0, np.pi, 100)
                e_angle = np.arctan2(e_vec[1], e_vec[0])
                ecc = abs(r_e - r_p) / (r_e + r_p)
                r_theta = (a_trans * (1 - ecc**2)) / (1 + ecc * np.cos(theta))
                ax1.plot(r_theta*np.cos(theta+e_angle), r_theta*np.sin(theta+e_angle), 'w--', alpha=0.4, label="Transfer Path")
                
                ax1.set_title(f"Solar System: Hohmann Path to {target_name}\nTotal Delta-V: {dv:.3f} km/s", fontsize=11)
                limit = max(r_p * 1.3, 2.5)
                ax1.set_xlim(-limit, limit); ax1.set_ylim(-limit, limit); ax1.set_aspect('equal')

                # --- Plot 2: L2 Point Analysis ---
                if p_input in MASS_RATIOS:
                    l2_dist_au = r_p * (MASS_RATIOS[p_input] / 3)**(1/3)
                    l2_dist_km = l2_dist_au * AU_KM
                    l2_vec = p_vec + (p_vec / r_p * l2_dist_au)
                    
                    ax2.scatter(p_vec[0], p_vec[1], s=300, color='lime', label=f"{target_name} Center")
                    ax2.scatter(l2_vec[0], l2_vec[1], s=150, color='red', marker='x', label='L2 Point')
                    ax2.plot([p_vec[0], l2_vec[0]], [p_vec[1], l2_vec[1]], 'w:', alpha=0.6)
                    
                    # Distance Label on Plot 2
                    ax2.text((p_vec[0]+l2_vec[0])/2, (p_vec[1]+l2_vec[1])/2, 
                             f" {l2_dist_km:,.0f} km", color='tomato', fontweight='bold', fontsize=10)
                    
                    ax2.set_title(f"L2 Zoom: {target_name}")
                    margin = l2_dist_au * 2.0
                    ax2.set_xlim(p_vec[0]-margin, p_vec[0]+margin)
                    ax2.set_ylim(p_vec[1]-margin, p_vec[1]+margin)
                    ax2.set_aspect('equal')
                    ax2.legend(loc='upper right', fontsize='small')
                    
                    fig.suptitle(f"Mission Analysis: Earth to {target_name} ({time_label})\nL2 Distance: {l2_dist_km:,.2f} km", color='white', y=0.96)
                else:
                    ax2.text(0.5, 0.5, "L2 mass data unavailable", ha='center')

            if time_choice == 'y':
                ani = animation.FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
                plt.show()
            else:
                update(0); plt.show()

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_app()
