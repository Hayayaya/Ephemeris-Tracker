import matplotlib
matplotlib.use('TkAgg') 

from skyfield.api import load, Topos
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import geonamescache
from timezonefinder import TimezoneFinder

# --- CONFIG & DATA ---
gc = geonamescache.GeonamesCache()
tf = TimezoneFinder()
ALL_CITIES = gc.get_cities()
EPHEMERIS_FILE = 'de422.bsp'
MU_SUN = 1.32712440018e11 
AU_KM = 149597870.7

ts = load.timescale(builtin=True)
planets = load(EPHEMERIS_FILE)
sun, earth = planets['sun'], planets['earth']

MASS_RATIOS = {'mercury': 1.66e-7, 'venus': 2.44e-6, 'earth': 3.00e-6, 'mars': 3.22e-7, 
               'jupiter': 9.54e-4, 'saturn': 2.85e-4, 'uranus': 4.36e-5, 'neptune': 5.15e-5}

def get_offline_location_data(city_query):
    search_name = city_query.title().strip()
    matches = [c for c in ALL_CITIES.values() if c['name'] == search_name]
    if not matches: return 51.4769, 0.0005, "UTC"
    target = max(matches, key=lambda x: x['population'])
    return float(target['latitude']), float(target['longitude']), tf.timezone_at(lng=float(target['longitude']), lat=float(target['latitude'])) or "UTC"

def get_phase_angle(vec1, vec2):
    return (np.degrees(np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0]))) % 360

def calculate_ideal_phase(r1, r2):
    return (180 * (1 - (1 / (2 * np.sqrt(2))) * (1 + r1 / r2)**1.5)) % 360

def calculate_deltav_split(r1_au, r2_au):
    r1, r2 = r1_au * AU_KM, r2_au * AU_KM
    v1 = np.sqrt(MU_SUN / r1)
    v_per = v1 * np.sqrt((2 * r2) / (r1 + r2))
    dv1 = abs(v_per - v1)
    v2 = np.sqrt(MU_SUN / r2)
    v_ap = v2 * np.sqrt((2 * r1) / (r1 + r2))
    dv2 = abs(v2 - v_ap)
    return dv1, dv2

def find_precise_window(target_name, start_dt, ideal_phi, seeking_open=True):
    for day in range(1, 1500):
        test_dt = start_dt + timedelta(days=day)
        t = ts.from_datetime(test_dt.replace(tzinfo=pytz.UTC))
        e = sun.at(t).observe(planets['earth barycenter']).position.au
        p = sun.at(t).observe(planets[target_name + ' barycenter']).position.au
        diff = abs(get_phase_angle(e, p) - ideal_phi)
        if (seeking_open and diff < 5.0) or (not seeking_open and diff > 5.0):
            return test_dt
    return None

def on_press(event):
    if event.key in ['q', 'escape']:
        plt.close('all')

def run_app():
    while True:
        print("\n" + "="*45 + "\n   ADVANCED MISSION ANALYZER V2\n" + "="*45)
        print(" (Type 'exit' to quit at any time)")
        
        t_choice = input("Use real-time tracking (y/n)? ").strip().lower()
        if t_choice == 'exit': break
        
        d_i, t_i = None, None
        if t_choice != 'y':
            d_i = input("Date (YYYY-MM-DD): ").strip()
            if d_i == 'exit': break
            t_i = input("Time (HH:MM): ").strip()
            if t_i == 'exit': break
        
        p_input = input("Target Planet: ").strip().lower()
        if p_input == 'exit': break
        loc_input = input("City Name: ").strip()
        if loc_input == 'exit': break

        try:
            lat, lon, tz_str = get_offline_location_data(loc_input)
            local_tz = pytz.timezone(tz_str)
            user_location = earth + Topos(latitude_degrees=lat, longitude_degrees=lon)
            
            plt.style.use('dark_background')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
            fig.canvas.mpl_connect('key_press_event', on_press)
            plt.subplots_adjust(wspace=0.3, bottom=0.2)
            
            def update(frame):
                if not plt.fignum_exists(fig.number): return
                ax1.clear(); ax2.clear(); ax3.clear()
                
                target_name = p_input.capitalize()
                target_planet = planets[p_input + ' barycenter']
                dt = datetime.now(local_tz) if t_choice == 'y' else local_tz.localize(datetime.strptime(f"{d_i} {t_i}", "%Y-%m-%d %H:%M"))
                t_curr = ts.from_datetime(dt)
                
                # Positions
                e_pos = sun.at(t_curr).observe(planets['earth barycenter']).position.au
                p_pos = sun.at(t_curr).observe(target_planet).position.au
                r_e, r_p = np.linalg.norm(e_pos), np.linalg.norm(p_pos)
                
                astrometric = user_location.at(t_curr).observe(target_planet)
                user_dist_km = astrometric.distance().km
                
                phi = get_phase_angle(e_pos, p_pos)
                ideal = calculate_ideal_phase(r_e, r_p)
                window_open = abs(phi - ideal) < 5.0
                dv1, dv2 = calculate_deltav_split(r_e, r_p)
                
                # --- PLOT 1: COMPLETE MISSION PLOT ---
                ax1.scatter(0, 0, color='yellow', s=100, edgecolors='orange') 
                theta = np.linspace(0, 2*np.pi, 200)
                ax1.plot(r_e * np.cos(theta), r_e * np.sin(theta), color='cyan', alpha=0.2)
                ax1.plot(r_p * np.cos(theta), r_p * np.sin(theta), color='lime', alpha=0.2)
                
                # Planets
                ax1.scatter(e_pos[0], e_pos[1], color='cyan', s=60)
                ax1.text(e_pos[0], e_pos[1], ' Earth', color='cyan', fontsize=9)
                ax1.scatter(p_pos[0], p_pos[1], color='lime', s=60)
                ax1.text(p_pos[0], p_pos[1], f' {target_name}', color='lime', fontsize=9)
                
                # Hohmann Transfer Arc
                h_theta = np.linspace(0, np.pi, 100)
                e_angle = np.arctan2(e_pos[1], e_pos[0])
                a_t = (r_e + r_p) / 2
                ecc = abs(r_e - r_p) / (r_e + r_p)
                r_h = (a_t * (1 - ecc**2)) / (1 + ecc * np.cos(h_theta))
                offset = np.pi if r_e > r_p else 0
                h_color = 'lime' if window_open else 'red'
                ax1.plot(r_h*np.cos(h_theta+e_angle+offset), r_h*np.sin(h_theta+e_angle+offset), 
                         color=h_color, linestyle='--', linewidth=2)
                
                # Satellite Capture Orbit (Shifted to Plot 1)
                if p_input in MASS_RATIOS:
                    l2d_au = r_p * (MASS_RATIOS[p_input] / 3)**(1/3)
                    sat_orbit_r = l2d_au * 0.5 # Visualization scaling
                    ax1.plot(p_pos[0] + sat_orbit_r*np.cos(theta), p_pos[1] + sat_orbit_r*np.sin(theta), 
                             color='white', linestyle='--', alpha=0.6, linewidth=1)
                    # Satellite Position on its orbit
                    ax1.scatter(p_pos[0] + sat_orbit_r * np.cos(frame/10), p_pos[1] + sat_orbit_r * np.sin(frame/10), 
                                color='white', s=15)

                ax1.set_title("Complete Mission Profile", fontsize=10)
                ax1.set_xlabel(f"Total Δv: {dv1+dv2:.2f} km/s", color='yellow')
                lim = max(r_p, r_e) * 1.3
                ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_aspect('equal')

                # --- PLOT 2: TARGET PROXIMITY & L2 ---
                if p_input in MASS_RATIOS:
                    l2d_au = r_p * (MASS_RATIOS[p_input] / 3)**(1/3)
                    l2v = p_pos + (p_pos / r_p * l2d_au)
                    ax2.scatter(p_pos[0], p_pos[1], s=250, color='lime')
                    
                    # L2 Indicator & Distance
                    ax2.scatter(l2v[0], l2v[1], s=100, color='red', marker='x')
                    ax2.plot([p_pos[0], l2v[0]], [p_pos[1], l2v[1]], 'w:', alpha=0.6)
                    ax2.text((p_pos[0]+l2v[0])/2, (p_pos[1]+l2v[1])/2, f" {l2d_au*AU_KM:,.0f} km", color='white', fontsize=8)
                    
                    ax2.set_title(f"{target_name} & L2 Zoom", fontsize=10)
                    m = l2d_au * 2.5
                    ax2.set_xlim(p_pos[0]-m, p_pos[0]+m); ax2.set_ylim(p_pos[1]-m, p_pos[1]+m); ax2.set_aspect('equal')

                # --- PLOT 3: DASHBOARD ---
                ax3.axis('off')
                next_win = find_precise_window(p_input, dt, ideal, not window_open)
                dash = (f"MISSION DASHBOARD\n{'='*25}\n"
                        f"Target: {target_name.upper()}\n"
                        f"DIST TO USER:\n{user_dist_km:,.0f} km\n\n"
                        f"BURN DATA:\n"
                        f"dv1 (Inject): {dv1:.2f} km/s\n"
                        f"dv2 (Capture): {dv2:.2f} km/s\n\n"
                        f"STATUS: {'WINDOW OPEN' if window_open else 'CLOSED'}\n")
                
                if window_open:
                    dash += f"CLOSES: {next_win.strftime('%Y-%m-%d') if next_win else 'N/A'}\n"
                else:
                    dash += f"OPENS:  {next_win.strftime('%Y-%m-%d') if next_win else 'N/A'}\n"
                
                dash += f"Phase: {phi:.1f}° (Goal: {ideal:.1f}°)"
                
                ax3.text(0, 0.5, dash, family='monospace', color='white', fontsize=10,
                         bbox=dict(facecolor='#111111', alpha=0.8, edgecolor=h_color, pad=12))
                fig.suptitle(f"System Analysis: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n(Press 'q' to return to menu)", color='white')

            if t_choice == 'y':
                ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
                plt.show()
            else:
                update(0); plt.show()

        except Exception as e:
            print(f"Error: {e}")
            
    print("Program Terminated.")

if __name__ == "__main__":
    run_app()
