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
MU_SUN = 1.32712440018e11 # km^3/s^2
AU_KM = 149597870.7

ts = load.timescale(builtin=True)
planets = load(EPHEMERIS_FILE)
sun = planets['sun']
earth = planets['earth']

MASS_RATIOS = {'mercury': 1.66e-7, 'venus': 2.44e-6, 'earth': 3.00e-6, 'mars': 3.22e-7, 
               'jupiter': 9.54e-4, 'saturn': 2.85e-4, 'uranus': 4.36e-5, 'neptune': 5.15e-5}

def get_offline_location_data(city_query):
    search_name = city_query.title().strip()
    matches = [c for c in ALL_CITIES.values() if c['name'] == search_name]
    if not matches: return 51.4769, 0.0005, "UTC"
    target = max(matches, key=lambda x: x['population'])
    return float(target['latitude']), float(target['longitude']), tf.timezone_at(lng=float(target['longitude']), lat=float(target['latitude'])) or "UTC"

def get_phase_angle(vec1, vec2):
    angle = np.degrees(np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0]))
    return angle % 360

def calculate_ideal_phase(r1, r2):
    return (180 * (1 - (1 / (2 * np.sqrt(2))) * (1 + r1 / r2)**1.5)) % 360

def calculate_deltav(r1_au, r2_au):
    r1, r2 = r1_au * AU_KM, r2_au * AU_KM
    v1 = np.sqrt(MU_SUN / r1)
    v_per = np.sqrt(MU_SUN / r1) * np.sqrt((2 * r2) / (r1 + r2))
    dv1 = abs(v_per - v1)
    v_ap = np.sqrt(MU_SUN / r2) * np.sqrt((2 * r1) / (r1 + r2))
    v2 = np.sqrt(MU_SUN / r2)
    dv2 = abs(v2 - v_ap)
    return dv1 + dv2

def find_precise_window(target_name, start_dt, ideal_phi, seeking_open=True):
    for day in range(1, 1500):
        test_dt = start_dt + timedelta(days=day)
        t = ts.from_datetime(test_dt.replace(tzinfo=pytz.UTC))
        e = sun.at(t).observe(planets['earth barycenter']).position.au
        p = sun.at(t).observe(planets[target_name + ' barycenter']).position.au
        diff = abs(get_phase_angle(e, p) - ideal_phi)
        if (seeking_open and diff < 5.0) or (not seeking_open and diff > 5.0):
            for hour in range(-24, 24):
                fine_dt = test_dt + timedelta(hours=hour)
                t_f = ts.from_datetime(fine_dt.replace(tzinfo=pytz.UTC))
                e_f = sun.at(t_f).observe(planets['earth barycenter']).position.au
                p_f = sun.at(t_f).observe(planets[target_name + ' barycenter']).position.au
                if (seeking_open and abs(get_phase_angle(e_f, p_f) - ideal_phi) < 5.0) or \
                   (not seeking_open and abs(get_phase_angle(e_f, p_f) - ideal_phi) > 5.0):
                    return fine_dt
    return None

def run_app():
    while True:
        print("\n" + "="*45 + "\n   ADVANCED MISSION ANALYZER\n" + "="*45)
        t_choice = input("Use real-time tracking (y/n)? ").strip().lower()
        if t_choice == 'exit': break
        
        d_i, t_i = None, None
        if t_choice != 'y':
            d_i = input("Date (YYYY-MM-DD): ").strip()
            t_i = input("Time (HH:MM): ").strip()
        
        p_input = input("Target Planet: ").strip().lower()
        loc_input = input("City Name: ").strip()

        try:
            lat, lon, tz_str = get_offline_location_data(loc_input)
            local_tz = pytz.timezone(tz_str)
            user_location = earth + Topos(latitude_degrees=lat, longitude_degrees=lon)
            
            plt.style.use('dark_background')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
            plt.subplots_adjust(wspace=0.3, bottom=0.2)
            
            def update(frame):
                ax1.clear(); ax2.clear(); ax3.clear()
                target_name = p_input.capitalize()
                target_planet = planets[p_input + ' barycenter']
                
                dt = datetime.now(local_tz) if t_choice == 'y' else local_tz.localize(datetime.strptime(f"{d_i} {t_i}", "%Y-%m-%d %H:%M"))
                t_curr = ts.from_datetime(dt)
                
                # Heliocentric Positions (for orbits)
                e_pos = sun.at(t_curr).observe(planets['earth barycenter']).position.au
                p_pos = sun.at(t_curr).observe(target_planet).position.au
                r_e, r_p = np.linalg.norm(e_pos), np.linalg.norm(p_pos)
                
                # Topocentric Distance (User to Planet)
                astrometric = user_location.at(t_curr).observe(target_planet)
                user_dist_km = astrometric.distance().km
                
                phi = get_phase_angle(e_pos, p_pos)
                ideal = calculate_ideal_phase(r_e, r_p)
                window_open = abs(phi - ideal) < 5.0
                dv_total = calculate_deltav(r_e, r_p)
                
                # --- PLOT 1: ORBITS ---
                ax1.scatter(0, 0, color='yellow', s=100, edgecolors='orange') 
                theta = np.linspace(0, 2*np.pi, 200)
                ax1.plot(r_e * np.cos(theta), r_e * np.sin(theta), color='cyan', alpha=0.3)
                ax1.plot(r_p * np.cos(theta), r_p * np.sin(theta), color='lime', alpha=0.3)
                ax1.scatter(e_pos[0], e_pos[1], color='cyan', s=60)
                ax1.scatter(p_pos[0], p_pos[1], color='lime', s=60)
                
                h_theta = np.linspace(0, np.pi, 100)
                e_angle = np.arctan2(e_pos[1], e_pos[0])
                a_t = (r_e + r_p) / 2
                ecc = abs(r_e - r_p) / (r_e + r_p)
                r_h = (a_t * (1 - ecc**2)) / (1 + ecc * np.cos(h_theta))
                offset = np.pi if r_e > r_p else 0
                h_color = 'lime' if window_open else 'red'
                ax1.plot(r_h*np.cos(h_theta+e_angle+offset), r_h*np.sin(h_theta+e_angle+offset), 
                         color=h_color, linestyle='--', linewidth=2)
                
                ax1.set_title("Hohmann Alignment", fontsize=10)
                ax1.set_xlabel(f"Transfer Δv: {dv_total:.2f} km/s", color='yellow')
                lim = max(r_p * 1.3, 2.5)
                ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_aspect('equal')

                # --- PLOT 2: L2 ZOOM ---
                if p_input in MASS_RATIOS:
                    l2d_au = r_p * (MASS_RATIOS[p_input] / 3)**(1/3)
                    l2v = p_pos + (p_pos / r_p * l2d_au)
                    ax2.scatter(p_pos[0], p_pos[1], s=200, color='lime')
                    ax2.scatter(l2v[0], l2v[1], s=100, color='red', marker='x')
                    ax2.plot([p_pos[0], l2v[0]], [p_pos[1], l2v[1]], 'w:', alpha=0.6)
                    ax2.text((p_pos[0]+l2v[0])/2, (p_pos[1]+l2v[1])/2, f" {l2d_au*AU_KM:,.0f} km", color='white', fontsize=8)
                    ax2.set_title(f"L2 Zoom", fontsize=10)
                    m = l2d_au * 3.0
                    ax2.set_xlim(p_pos[0]-m, p_pos[0]+m); ax2.set_ylim(p_pos[1]-m, p_pos[1]+m); ax2.set_aspect('equal')

                # --- PLOT 3: DASHBOARD (Target to User Dist) ---
                ax3.axis('off')
                next_start = find_precise_window(p_input, dt, ideal, True)
                win_exit = find_precise_window(p_input, dt, ideal, False) if window_open else None
                dash = (f"MISSION DASHBOARD\n{'='*25}\n"
                        f"Target: {target_name.upper()}\n"
                        f"User Loc: {loc_input.title()}\n"
                        f"DIST TO USER:\n{user_dist_km:,.2f} km\n\n"
                        f"Phase: {phi:.2f}° (Goal: {ideal:.2f}°)\n"
                        f"STATUS: {'OPEN' if window_open else 'CLOSED'}\n")
                if window_open:
                    dash += f"CLOSES: {win_exit.strftime('%Y-%m-%d %H:%M') if win_exit else 'N/A'}"
                else:
                    dash += f"OPENS:  {next_start.strftime('%Y-%m-%d %H:%M') if next_start else 'N/A'}"
                
                ax3.text(0, 0.5, dash, family='monospace', color='white', fontsize=10,
                         bbox=dict(facecolor='#111111', alpha=0.8, edgecolor=h_color, pad=12))
                fig.suptitle(f"Live Analysis: {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}", color='white', fontweight='bold')

            if t_choice == 'y':
                ani = animation.FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
                plt.show()
            else:
                update(0); plt.show()

        except Exception as e: print(f"Error: {e}")
            
    print("Exiting...")

if __name__ == "__main__":
    run_app()
