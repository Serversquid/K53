"""
=============================================================
  GLOSA - Green Light Optimal Speed Advisory (KÄMIL VERSIÝA)
  Awtoulag Sürüjilerine Swetafor Maslahat Ulgamy
  ─────────────────────────────────────────────
  TÄZE GOŞUNDYLAR:
    1. Köp Swetaforly Ulgam   – birnäçe swetafor zynjyry
    2. Howa Şertleri Modeli   – gar, ýagyş, duman täsiri
    3. ML Prediksiýa Modeli   – Random Forest bilen sikl çaklamak
=============================================================
  Diplom işi : Şanazar Sarjaýew
  Magtymguly adyndaky TDU, 2026
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
#  1. PARAMETRLER (KONFIGURASIÝA)
# ═══════════════════════════════════════════════════════════
class Config:
    # Swetafor
    GREEN_MIN   = 25
    GREEN_MAX   = 60
    RED_MIN     = 25
    RED_MAX     = 60

    # Ulag tizligi
    V_MIN_KMH   = 30
    V_MAX_KMH   = 90
    V_MIN       = V_MIN_KMH / 3.6
    V_MAX       = V_MAX_KMH / 3.6

    # Köp swetafor ulgamy
    N_INTERSECTIONS  = 4          # swetafor sany
    INTERSECTION_GAP = 300        # swetaforlar arasyndaky aralyk (m)

    # Ýangyç
    A_FUEL      = 0.05
    B_FUEL      = 0.003
    IDLE_FUEL   = 0.5

    # Simulýasiýa
    N_SIMS      = 1000
    DT          = 0.1

    # Howa täsiriniň koefisiýentleri
    WEATHER_FRICTION = {
        'clear':  1.00,   # adaty
        'rain':   0.82,   # ýagyş → tizlik azalýar
        'snow':   0.62,   # gar → has haýal
        'fog':    0.74,   # duman → görüşlik az
    }
    WEATHER_FUEL_COEF = {
        'clear': 1.00,
        'rain':  1.12,
        'snow':  1.28,
        'fog':   1.08,
    }
    WEATHER_PROB = {
        'clear': 0.60,
        'rain':  0.20,
        'snow':  0.12,
        'fog':   0.08,
    }

# ═══════════════════════════════════════════════════════════
#  2. HOWA ŞERTLERI MODELI
# ═══════════════════════════════════════════════════════════
class WeatherCondition:
    CONDITIONS = ['clear', 'rain', 'snow', 'fog']
    LABELS = {
        'clear': '☀️ Açyk',
        'rain':  '🌧️ Ýagyş',
        'snow':  '❄️ Gar',
        'fog':   '🌫️ Duman',
    }

    def __init__(self, condition=None):
        if condition:
            self.condition = condition
        else:
            probs  = list(Config.WEATHER_PROB.values())
            self.condition = np.random.choice(self.CONDITIONS, p=probs)

    @property
    def friction(self):
        """Tizlik azalma koef. (1.0 = adaty)"""
        return Config.WEATHER_FRICTION[self.condition]

    @property
    def fuel_coef(self):
        """Ýangyç artma koef. (1.0 = adaty)"""
        return Config.WEATHER_FUEL_COEF[self.condition]

    @property
    def v_max(self):
        """Howa şertine görä iň ýokary tizlik"""
        return Config.V_MAX * self.friction

    @property
    def label(self):
        return self.LABELS[self.condition]

    def visibility_penalty(self):
        """Görüşlik azalmasy: goşmaça howpsuzlyk germewi (metr görnüşinde)"""
        penalties = {'clear': 0, 'rain': 8, 'snow': 15, 'fog': 20}
        return penalties[self.condition]   # metr

# ═══════════════════════════════════════════════════════════
#  3. SWETAFOR (KÄMIL VERSIÝA)
# ═══════════════════════════════════════════════════════════
class TrafficLight:
    def __init__(self, green_dur=None, red_dur=None, position=0):
        self.green_dur = green_dur or np.random.randint(
            Config.GREEN_MIN, Config.GREEN_MAX + 1)
        self.red_dur   = red_dur   or np.random.randint(
            Config.RED_MIN,   Config.RED_MAX   + 1)
        self.cycle     = self.green_dur + self.red_dur
        self.phase     = np.random.uniform(0, self.cycle)
        self.position  = position    # ýolda ýerleşen ýeri (m)

        # ML model üçin synlama taryhy
        self._history  = []

    def get_state(self, t):
        current = (self.phase + t) % self.cycle
        if current < self.green_dur:
            return 'green', self.green_dur - current
        else:
            return 'red', self.cycle - current

    def time_to_green(self, t):
        state, remaining = self.get_state(t)
        return 0 if state == 'green' else remaining

    def time_to_red(self, t):
        state, remaining = self.get_state(t)
        return 0 if state == 'red' else remaining

    def record_observation(self, t, state):
        """ML model üçin synlama ýazgysy"""
        self._history.append({
            't': t,
            'state': 1 if state == 'green' else 0,
            'phase_in_cycle': (self.phase + t) % self.cycle,
        })

# ═══════════════════════════════════════════════════════════
#  4. KÖP SWETAFORLY ULGAM
# ═══════════════════════════════════════════════════════════
class MultiIntersectionNetwork:
    """
    N sany swetafory yzygiderli zynjyr görnüşinde gurnap berýär.
    Her swetafor biri-birinden INTERSECTION_GAP metr uzakda.
    """
    def __init__(self, n=Config.N_INTERSECTIONS):
        self.n = n
        self.intersections = []
        for i in range(n):
            pos = (i + 1) * Config.INTERSECTION_GAP
            # Utgaşdyrylan ýaşyl tolkuny üçin faz tapawudy
            phase_offset = i * (Config.GREEN_MIN + Config.RED_MIN) / n
            tl = TrafficLight(position=pos)
            tl.phase = (tl.phase + phase_offset) % tl.cycle
            self.intersections.append(tl)

    @property
    def total_length(self):
        return (self.n + 1) * Config.INTERSECTION_GAP

    def next_intersection(self, x):
        """Häzirki pozisiýadan öňdäki ilkinji swetafory tapýar"""
        for tl in self.intersections:
            if tl.position > x:
                return tl, tl.position - x
        return None, None

# ═══════════════════════════════════════════════════════════
#  5. ML PREDIKSIÝA MODELI (Random Forest)
# ═══════════════════════════════════════════════════════════
class GLOSAMLPredictor:
    """
    Random Forest bilen swetafor galan wagtyny çaklaýar.
    Öwrenme: geçen tizlik, faz, howa, ýol
    Çykyş: ýaşyla çenli galan wagt (sek)
    """
    def __init__(self):
        self.model   = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.scaler  = StandardScaler()
        self.trained = False
        self._mae    = None
        self._r2     = None

    def _generate_training_data(self, n_samples=5000):
        """Emeli taýýarlyk maglumatlary döredýär"""
        X, y = [], []
        weather_codes = {'clear': 0, 'rain': 1, 'snow': 2, 'fog': 3}

        for _ in range(n_samples):
            green_dur = np.random.randint(Config.GREEN_MIN, Config.GREEN_MAX + 1)
            red_dur   = np.random.randint(Config.RED_MIN,   Config.RED_MAX   + 1)
            cycle     = green_dur + red_dur
            t_now     = np.random.uniform(0, cycle * 3)
            phase     = np.random.uniform(0, cycle)
            current   = (phase + t_now) % cycle

            # Häzirki ýagdaý
            if current < green_dur:
                state = 1
                time_to_red = green_dur - current
                time_to_green = 0
            else:
                state = 0
                time_to_red = 0
                time_to_green = cycle - current

            v         = np.random.uniform(Config.V_MIN, Config.V_MAX)
            distance  = np.random.uniform(50, 500)
            weather   = np.random.choice(list(weather_codes.keys()),
                        p=list(Config.WEATHER_PROB.values()))
            w_code    = weather_codes[weather]
            friction  = Config.WEATHER_FRICTION[weather]

            # Öwreniş alamatlary (features)
            features = [
                state,                    # häzirki ýagdaý
                current / cycle,          # sikl içindäki faz (normalizasiýa)
                green_dur / cycle,        # ýaşyl proporsion
                red_dur / cycle,          # gyzyl proporsion
                v / Config.V_MAX,         # normalizasiýa edilmiş tizlik
                distance / 500,           # normalizasiýa edilmiş aralyk
                w_code / 3,               # howa kody
                friction,                 # sürtünme
                np.sin(2 * np.pi * current / cycle),   # sikl trigonometriýasy
                np.cos(2 * np.pi * current / cycle),
            ]

            # Çykyş: ýaşyla çenli galan wagt
            target = time_to_green if state == 0 else (red_dur + time_to_green)
            X.append(features)
            y.append(target)

        return np.array(X), np.array(y)

    def train(self):
        """Modeli taýýarlyk maglumatlary bilen tälim berýär"""
        print("  🤖 ML modeli öwredilýär...", end='', flush=True)
        X, y = self._generate_training_data(n_samples=8000)

        # 80/20 böl
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        X_tr_sc = self.scaler.fit_transform(X_tr)
        X_te_sc = self.scaler.transform(X_te)

        self.model.fit(X_tr_sc, y_tr)
        y_pred = self.model.predict(X_te_sc)

        self._mae = mean_absolute_error(y_te, y_pred)
        self._r2  = r2_score(y_te, y_pred)
        self.trained = True
        print(f" tamamlandy! MAE={self._mae:.2f}s  R²={self._r2:.3f}")

    def predict_time_to_green(self, tl, t_now, v, distance, weather):
        """Real wagtda galan ýaşyl wagty çaklaýar"""
        if not self.trained:
            # Öwretmedik bolsa, analitik hasap
            return tl.time_to_green(t_now)

        weather_codes = {'clear': 0, 'rain': 1, 'snow': 2, 'fog': 3}
        state_code, _ = tl.get_state(t_now)
        current = (tl.phase + t_now) % tl.cycle

        features = np.array([[
            1 if state_code == 'green' else 0,
            current / tl.cycle,
            tl.green_dur / tl.cycle,
            tl.red_dur   / tl.cycle,
            v / Config.V_MAX,
            distance / 500,
            weather_codes.get(weather.condition, 0) / 3,
            weather.friction,
            np.sin(2 * np.pi * current / tl.cycle),
            np.cos(2 * np.pi * current / tl.cycle),
        ]])

        features_sc = self.scaler.transform(features)
        pred = float(self.model.predict(features_sc)[0])
        return max(0, pred)

# ═══════════════════════════════════════════════════════════
#  6. ÝANGYÇ HASAPLAMASY
# ═══════════════════════════════════════════════════════════
def fuel_rate(v_ms, weather: WeatherCondition):
    v_kmh = v_ms * 3.6
    f_per_100km = Config.A_FUEL * v_kmh + Config.B_FUEL * v_kmh**2
    f_per_km    = f_per_100km / 100.0
    f_per_sec   = f_per_km * v_ms / 1000.0
    return max(f_per_sec, 0) * weather.fuel_coef

def idle_fuel_per_sec():
    return Config.IDLE_FUEL / 3600.0

# ═══════════════════════════════════════════════════════════
#  7. GLOSA ALGORITMI (KÄMIL – ML + Köp Swetafor + Howa)
# ═══════════════════════════════════════════════════════════
def glosa_advisory_advanced(distance, tl, t_now, v_current, weather,
                             ml_predictor=None):
    """
    Kämil GLOSA: ML çaklamasy + howa şertleri + howpsuzlyk margini
    """
    state, remaining = tl.get_state(t_now)
    visibility_margin = weather.visibility_penalty()    # görüşlik margini

    # ML ýa-da analitik çaklama
    if ml_predictor and ml_predictor.trained:
        ttg = ml_predictor.predict_time_to_green(tl, t_now, v_current,
                                                  distance, weather)
    else:
        ttg = tl.time_to_green(t_now)

    v_max_weather = weather.v_max    # howa şertine görä ýokary tizlik

    if v_current > 0:
        t_arrive = distance / v_current
    else:
        t_arrive = float('inf')

    if state == 'green':
        t_red = tl.time_to_red(t_now)
        if t_arrive <= t_red - visibility_margin / max(v_current, 1):
            return v_current, 'green_wave', \
                f"✅ ÝAŞYL TOLKUNY [{weather.label}]: {v_current*3.6:.1f} km/s"
        else:
            t_next_green = t_red + tl.red_dur
            v_opt = distance / (t_next_green + visibility_margin / Config.V_MAX)
            v_opt = np.clip(v_opt, Config.V_MIN, v_max_weather)
            return v_opt, 'slow_down', \
                f"🟡 HAÝALLAŞYÑ [{weather.label}]: {v_opt*3.6:.1f} km/s"
    else:
        if t_arrive < ttg:
            v_opt = distance / (ttg + 5 + visibility_margin / Config.V_MAX)
            v_opt = np.clip(v_opt, Config.V_MIN, v_max_weather)
            return v_opt, 'slow_down', \
                f"🔴 GARAŞYÑ [{weather.label}]: {v_opt*3.6:.1f} km/s"
        else:
            v_opt = distance / max(ttg, 1)
            v_opt = np.clip(v_opt, Config.V_MIN, v_max_weather)
            return v_opt, 'green_wave', \
                f"✅ ML ÇAKLAMA [{weather.label}]: {v_opt*3.6:.1f} km/s"

# ═══════════════════════════════════════════════════════════
#  8. KÖP SWETAFORLY SIMULÝASIÝA
# ═══════════════════════════════════════════════════════════
def run_simulation_multi(network: MultiIntersectionNetwork,
                          v_init, weather: WeatherCondition,
                          ml_predictor=None, use_glosa=True):
    """
    Köp swetaforly yzygider ýolda simulýasiýa.
    Ulag her swetaforda GLOSA maslahatyny alýar.
    """
    x         = 0.0
    v         = min(v_init, weather.v_max)
    t         = 0.0
    fuel      = 0.0
    stops     = 0
    total_stop_time = 0.0

    v_profile = [v * 3.6]
    t_profile = [0.0]
    x_profile = [0.0]

    total_dist = network.total_length

    while x < total_dist:
        # Öňdäki swetafory tap
        tl, dist_to_tl = network.next_intersection(x)

        if tl is None or dist_to_tl is None:
            # Ähli swetafordan geçildi — erkin sürmek dowam edilýär
            remaining = total_dist - x
            if v > 0:
                t_finish = remaining / v
                fuel += fuel_rate(v, weather) * t_finish
                t    += t_finish
            x = total_dist
            break
        else:
            # GLOSA maslahat (toplu simulýasiýada analitik usul ulanylýar)
            if use_glosa and dist_to_tl < 350:
                v_rec, strategy, _ = glosa_advisory_advanced(
                    dist_to_tl, tl, t, v, weather, ml_predictor=None)
                accel = np.clip((v_rec - v) / 3.0, -3.0, 2.0)
            else:
                accel = 0.0

            v = np.clip(v + accel * Config.DT, 0, weather.v_max)
            state, _ = tl.get_state(t)
            dist_step = max(v * Config.DT, 0.01)

            # Swetafor ýerinde barlag
            if dist_to_tl <= dist_step:
                if state == 'red':
                    stops += 1
                    wait   = tl.time_to_green(t)
                    fuel  += idle_fuel_per_sec() * wait
                    t     += wait
                    total_stop_time += wait
                    v      = Config.V_MIN
                x = tl.position
            else:
                x += dist_step

        fuel += fuel_rate(v, weather) * Config.DT
        t    += Config.DT
        v_profile.append(v * 3.6)
        t_profile.append(t)
        x_profile.append(x)

        if t > 400:
            break

    return {
        'fuel_total'   : fuel,
        'stops'        : stops,
        'stop_time'    : total_stop_time,
        'travel_time'  : t,
        'v_profile'    : np.array(v_profile),
        't_profile'    : np.array(t_profile),
        'x_profile'    : np.array(x_profile),
        'weather'      : weather.condition,
    }

# ═══════════════════════════════════════════════════════════
#  9. KÖPÇÜLIKLEÝIN SIMULÝASIÝA
# ═══════════════════════════════════════════════════════════
def run_batch_advanced(n=Config.N_SIMS, ml_predictor=None):
    print(f"\n{'='*60}")
    print(f"  KÄMIL GLOSA Simulýasiýasy  (N={n})")
    print(f"  Köp Swetafor | Howa Şertleri | ML Çaklama")
    print(f"{'='*60}")

    results_glosa    = []
    results_no_glosa = []

    for i in range(n):
        network = MultiIntersectionNetwork()
        v_init  = np.random.uniform(Config.V_MIN, Config.V_MAX)
        weather = WeatherCondition()

        rg  = run_simulation_multi(network, v_init, weather,
                                    ml_predictor=ml_predictor, use_glosa=True)
        rng = run_simulation_multi(network, v_init, weather,
                                    ml_predictor=ml_predictor, use_glosa=False)
        results_glosa.append(rg)
        results_no_glosa.append(rng)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n} simulýasiýa tamamlandy...")

    return results_glosa, results_no_glosa

# ═══════════════════════════════════════════════════════════
#  10. STATISTIKA
# ═══════════════════════════════════════════════════════════
def compute_stats_advanced(results_glosa, results_no_glosa):
    fuel_g  = np.array([r['fuel_total']  for r in results_glosa])
    fuel_ng = np.array([r['fuel_total']  for r in results_no_glosa])
    stops_g = np.array([r['stops']       for r in results_glosa],   dtype=float)
    stops_ng= np.array([r['stops']       for r in results_no_glosa],dtype=float)
    time_g  = np.array([r['travel_time'] for r in results_glosa])
    time_ng = np.array([r['travel_time'] for r in results_no_glosa])
    weathers= [r['weather']              for r in results_glosa]

    savings_pct = np.where(fuel_ng > 0, (fuel_ng - fuel_g) / fuel_ng * 100, 0)

    # Howa görä bölüp hasapla
    weather_savings = {}
    for cond in WeatherCondition.CONDITIONS:
        mask = np.array([w == cond for w in weathers])
        if mask.sum() > 0:
            weather_savings[cond] = savings_pct[mask].mean()

    return {
        'fuel_glosa_mean'    : fuel_g.mean(),
        'fuel_noglosa_mean'  : fuel_ng.mean(),
        'savings_mean_pct'   : savings_pct.mean(),
        'savings_median_pct' : np.median(savings_pct),
        'savings_pct_arr'    : savings_pct,
        'stops_glosa_mean'   : stops_g.mean(),
        'stops_noglosa_mean' : stops_ng.mean(),
        'stop_reduction_pct' : (stops_ng.mean() - stops_g.mean()) /
                                max(stops_ng.mean(), 0.001) * 100,
        'time_glosa_mean'    : time_g.mean(),
        'time_noglosa_mean'  : time_ng.mean(),
        'weather_savings'    : weather_savings,
        'fuel_g_arr'         : fuel_g,
        'fuel_ng_arr'        : fuel_ng,
        'stops_g_arr'        : stops_g,
        'stops_ng_arr'       : stops_ng,
        'weathers'           : weathers,
    }

# ═══════════════════════════════════════════════════════════
#  11. NETIJELER ÇYKARMA
# ═══════════════════════════════════════════════════════════
def print_results_advanced(stats, ml_predictor=None):
    print(f"\n{'='*60}")
    print(f"  📊 KÄMIL GLOSA – SIMULÝASIÝA NETIJELERI")
    print(f"{'='*60}")

    print(f"\n  🛣️  Ýol: {Config.N_INTERSECTIONS} swetafor, "
          f"{Config.N_INTERSECTIONS * Config.INTERSECTION_GAP}m")

    print(f"\n  ⛽ Ýangyç harçlamagy (ortalama):")
    print(f"    GLOSA ÝOK : {stats['fuel_noglosa_mean']*1000:.2f} ml")
    print(f"    GLOSA BAR : {stats['fuel_glosa_mean']*1000:.2f} ml")
    print(f"    Tygşytlylyk: %{stats['savings_mean_pct']:.1f}")

    print(f"\n  🚦 Säginme sany (ortalama):")
    print(f"    GLOSA ÝOK : {stats['stops_noglosa_mean']:.2f}")
    print(f"    GLOSA BAR : {stats['stops_glosa_mean']:.2f}")
    print(f"    Azalma    : %{stats['stop_reduction_pct']:.1f}")

    print(f"\n  ⏱️  Ýol wagty (ortalama):")
    print(f"    GLOSA ÝOK : {stats['time_noglosa_mean']:.1f} sek")
    print(f"    GLOSA BAR : {stats['time_glosa_mean']:.1f} sek")

    print(f"\n  🌦️  Howa görä tygşytlylyk:")
    for cond, sav in stats['weather_savings'].items():
        label = WeatherCondition.LABELS.get(cond, cond)
        print(f"    {label:<12} : %{sav:.1f}")

    if ml_predictor and ml_predictor.trained:
        print(f"\n  🤖 ML Model (Random Forest):")
        print(f"    MAE : {ml_predictor._mae:.2f} sek")
        print(f"    R²  : {ml_predictor._r2:.3f}")
        imps = ml_predictor.model.feature_importances_
        feat_names = ['Ýagdaý','Faz','Ýaşyl Prop.','Gyzyl Prop.',
                      'Tizlik','Aralyk','Howa Kody','Sürtünme','sin(faz)','cos(faz)']
        top = sorted(zip(feat_names, imps), key=lambda x: -x[1])[:3]
        print(f"    Iň möhüm belgiler: "
              + ", ".join(f"{n} ({v:.2f})" for n, v in top))

    print(f"\n{'='*60}\n")

# ═══════════════════════════════════════════════════════════
#  12. GÖRSEL (GRAFIK) – 6 panel
# ═══════════════════════════════════════════════════════════
def plot_results_advanced(stats, results_glosa, results_no_glosa,
                           ml_predictor=None):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    gs  = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.40)

    DARK  = '#0d1117'
    CARD  = '#161b22'
    GREEN = '#2ea043'
    RED   = '#da3633'
    BLUE  = '#1f6feb'
    GOLD  = '#e3b341'
    PURP  = '#8957e5'
    CYAN  = '#39d353'
    TEXT  = '#c9d1d9'
    MUTED = '#8b949e'

    WEATHER_COLORS = {
        'clear': GOLD, 'rain': BLUE, 'snow': CYAN, 'fog': MUTED
    }

    def style_ax(ax, title):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(title, color=TEXT, fontsize=9,
                     fontweight='bold', pad=8, fontfamily='monospace')

    # ── 1. Ýangyç tygşytlylygy (histogramma) ─────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "📊 Ýangyç Tygşytlylygy Paýlanyşy (N=1000)")
    s = stats['savings_pct_arr']
    ax1.hist(s[s > 0],  bins=45, color=GREEN, alpha=0.85,
             label='Tygşytlylyk', edgecolor='#30363d', lw=0.3)
    ax1.hist(s[s <= 0], bins=15, color=RED,   alpha=0.70,
             label='Artma', edgecolor='#30363d', lw=0.3)
    ax1.axvline(s.mean(), color=GOLD, ls='--', lw=2,
                label=f'Ort: %{s.mean():.1f}')
    ax1.set_xlabel("Tygşytlylyk (%)", color=MUTED)
    ax1.set_ylabel("Simulýasiýa sany", color=MUTED)
    ax1.legend(facecolor=CARD, edgecolor='#30363d', labelcolor=TEXT, fontsize=8)
    ax1.grid(axis='y', color='#21262d', lw=0.5)

    # ── 2. KPI karta ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(CARD)
    for sp in ax2.spines.values(): sp.set_color('#30363d')
    ax2.set_xticks([]); ax2.set_yticks([])
    kpis = [
        ("Ort. Tygşytlylyk",     f"%{stats['savings_mean_pct']:.1f}", GREEN),
        ("Säginme Azalmagy",     f"%{stats['stop_reduction_pct']:.1f}", RED),
        ("Ort. Säginme (GLOSA)", f"{stats['stops_glosa_mean']:.1f}",   BLUE),
        ("Wagt Tapawudy",        f"{stats['time_noglosa_mean']-stats['time_glosa_mean']:.1f}s", GOLD),
    ]
    for idx, (label, val, color) in enumerate(kpis):
        y = 0.82 - idx * 0.22
        ax2.text(0.5, y,      val,   ha='center', va='center',
                 color=color, fontsize=17, fontweight='bold',
                 transform=ax2.transAxes, fontfamily='monospace')
        ax2.text(0.5, y-0.08, label, ha='center', va='center',
                 color=MUTED, fontsize=8, transform=ax2.transAxes)
    ax2.set_title("🎯 Esasy Görkezijiler", color=TEXT, fontsize=9,
                  fontweight='bold', pad=8)

    # ── 3. Howa görä tygşytlylyk (Bar chart) ─────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "🌦️ Howa Şertine Görä Tygşytlylyk (%)")
    ws = stats['weather_savings']
    labels_w = [WeatherCondition.LABELS[c] for c in ws.keys()]
    values_w = list(ws.values())
    cols_w   = [WEATHER_COLORS[c] for c in ws.keys()]
    bars = ax3.bar(labels_w, values_w, color=cols_w, alpha=0.85,
                   edgecolor='#30363d', width=0.6)
    for bar, val in zip(bars, values_w):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f'%{val:.1f}', ha='center',
                 color=TEXT, fontsize=9, fontweight='bold')
    ax3.set_ylabel("Tygşytlylyk (%)", color=MUTED)
    ax3.set_ylim(0, max(values_w) * 1.35)
    ax3.grid(axis='y', color='#21262d', lw=0.5)
    plt.setp(ax3.get_xticklabels(), fontsize=7)

    # ── 4. Säginme sany deňeşdirmesi ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "🚦 Säginme Sany (Ort.)")
    cats  = ['GLOSA\nÝOK', 'GLOSA\nBAR']
    rates = [stats['stops_noglosa_mean'], stats['stops_glosa_mean']]
    cols  = [RED, GREEN]
    bars2 = ax4.bar(cats, rates, color=cols, alpha=0.85,
                    edgecolor='#30363d', width=0.5)
    for b, v_ in zip(bars2, rates):
        ax4.text(b.get_x() + b.get_width()/2,
                 b.get_height() + 0.05,
                 f'{v_:.2f}', ha='center',
                 color=TEXT, fontsize=11, fontweight='bold')
    ax4.set_ylabel("Säginme sany (ort.)", color=MUTED)
    ax4.set_ylim(0, max(rates) * 1.35)
    ax4.grid(axis='y', color='#21262d', lw=0.5)

    # ── 5. Ýangyç Box plot ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, "⛽ Ýangyç (ml) – Ýaý Diagrammasy")
    bp = ax5.boxplot(
        [stats['fuel_ng_arr']*1000, stats['fuel_g_arr']*1000],
        patch_artist=True, widths=0.5,
        medianprops=dict(color=GOLD, lw=2),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        flierprops=dict(marker='o', color=MUTED, ms=2, alpha=0.3)
    )
    bp['boxes'][0].set_facecolor(RED);   bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(GREEN); bp['boxes'][1].set_alpha(0.7)
    ax5.set_xticklabels(['GLOSA ÝOK', 'GLOSA BAR'], color=TEXT)
    ax5.set_ylabel("Ýangyç (ml)", color=MUTED)
    ax5.grid(axis='y', color='#21262d', lw=0.5)

    # ── 6. Tizlik profili (köp swetaforly) ───────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    style_ax(ax6, "🚗 Tizlik Profili – Köp Swetaforly Ýol (GLOSA BAR vs ÝOK)")
    idx_best = int(np.argmax(stats['savings_pct_arr']))
    rg   = results_glosa[idx_best]
    rng  = results_no_glosa[idx_best]
    cond = rg['weather']
    wlbl = WeatherCondition.LABELS.get(cond, cond)

    ax6.plot(rng['t_profile'], rng['v_profile'],
             color=RED,   lw=2, alpha=0.85, label='GLOSA ÝOK')
    ax6.plot(rg['t_profile'],  rg['v_profile'],
             color=GREEN, lw=2, alpha=0.85, label=f'GLOSA BAR [{wlbl}]')

    v_max_w = Config.V_MAX_KMH * Config.WEATHER_FRICTION[cond]
    ax6.axhline(v_max_w, color=WEATHER_COLORS.get(cond, GOLD),
                ls=':', lw=1.5, alpha=0.7,
                label=f'Howa Çägi: {v_max_w:.0f} km/s')
    ax6.axhline(Config.V_MIN_KMH, color=MUTED, ls=':', lw=1, alpha=0.5)

    # Swetafor ýerlerini bellik et
    for i in range(Config.N_INTERSECTIONS):
        tx = (i + 1) * Config.INTERSECTION_GAP
        # Takmyn wagtda swetafora ýetmek
        ax6.axvline(tx / max(rg['v_profile'].mean() / 3.6, 1),
                    color='#30363d', ls='--', lw=0.7, alpha=0.5)

    ax6.set_xlabel("Wagt (sek)", color=MUTED)
    ax6.set_ylabel("Tizlik (km/s)", color=MUTED)
    ax6.set_ylim(0, Config.V_MAX_KMH + 15)
    ax6.legend(facecolor=CARD, edgecolor='#30363d',
               labelcolor=TEXT, fontsize=9)
    ax6.grid(color='#21262d', lw=0.5)

    # ML bellik
    if ml_predictor and ml_predictor.trained:
        ax6.text(0.01, 0.95,
                 f"🤖 ML: MAE={ml_predictor._mae:.1f}s  R²={ml_predictor._r2:.3f}",
                 transform=ax6.transAxes, color=PURP,
                 fontsize=8, va='top', fontfamily='monospace')

    # Başlyk
    fig.suptitle(
        "GLOSA KÄMIL ULGAM – Köp Swetafor | Howa Şertleri | ML Çaklama\n"
        "Diplom Iş Simulýasiýasy  |  Şanazar Sarjaýew  |  TDU 2026",
        color=TEXT, fontsize=12, fontweight='bold', y=0.99,
        fontfamily='monospace'
    )

    plt.savefig('/mnt/user-data/outputs/glosa_advanced_results.png',
                dpi=150, bbox_inches='tight', facecolor=DARK)
    print("  📈 Grafik saklanan: glosa_advanced_results.png")
    plt.close()

# ═══════════════════════════════════════════════════════════
#  13. INTERAKTIW DEMO (täze şertler bilen)
# ═══════════════════════════════════════════════════════════
def interactive_demo_advanced(ml_predictor):
    print(f"\n{'='*60}")
    print(f"  🚗 KÄMIL GLOSA – INTERAKTIW DEMO")
    print(f"{'='*60}")

    scenarios = [
        (350, 70, 0,  'clear', "Açyk howa, uzak"),
        (180, 55, 18, 'rain',  "Ýagyşly, orta aralyk"),
        (250, 65, 38, 'snow',  "Garly, çalt barýar"),
        (120, 45, 44, 'fog',   "Dumanlyk, gaty ýakyn"),
    ]

    network = MultiIntersectionNetwork()
    tl = network.intersections[0]

    for dist, v_kmh, t_now, cond, note in scenarios:
        v_ms    = v_kmh / 3.6
        weather = WeatherCondition(cond)
        state, remaining = tl.get_state(t_now)
        v_rec, strategy, msg = glosa_advisory_advanced(
            dist, tl, t_now, v_ms, weather, ml_predictor)
        sym = "🟢" if state == 'green' else "🔴"

        print(f"\n  [{note}]")
        print(f"  Mesafe    : {dist} m  |  Tizlik: {v_kmh} km/s")
        print(f"  Howa      : {weather.label}  |  Tizlik çägi: {weather.v_max*3.6:.0f} km/s")
        print(f"  Swetafor  : {sym} ({remaining:.0f} sek galanda)")
        print(f"  Strategiýa: {strategy}")
        print(f"  Maslahat  : {msg}")
    print()

# ═══════════════════════════════════════════════════════════
#  14. BÖLEKLEÝIN SYNAGLAR
# ═══════════════════════════════════════════════════════════
def unit_tests_advanced():
    print(f"\n{'='*60}")
    print(f"  🧪 BÖLEKLEÝIN SYNAGLAR (KÄMIL)")
    print(f"{'='*60}")

    # Howa synagy
    for cond in WeatherCondition.CONDITIONS:
        w = WeatherCondition(cond)
        assert 0 < w.friction <= 1, f"{cond}: sürtünme ýalňyş"
        assert w.fuel_coef >= 1,    f"{cond}: ýangyç koef ýalňyş"
    print("  ✅ Howa şertleri: ähli koefisiýentler dogry")

    # Köp swetafor ulgamy synagy
    net = MultiIntersectionNetwork(4)
    assert len(net.intersections) == 4
    assert net.intersections[-1].position == 4 * Config.INTERSECTION_GAP
    print(f"  ✅ Köp swetafor ulgamy: {len(net.intersections)} swetafor, "
          f"jemi {net.total_length} m")

    # ML modeli baglanyşyk synagy
    ml = GLOSAMLPredictor()
    ml.train()
    assert ml.trained
    assert ml._r2 > 0.5, f"R² gaty pes: {ml._r2}"
    print(f"  ✅ ML model öwredildi: R²={ml._r2:.3f}")

    # GLOSA advanced synagy
    net2 = MultiIntersectionNetwork(1)
    tl2  = net2.intersections[0]
    w2   = WeatherCondition('rain')
    v_r, st, msg = glosa_advisory_advanced(200, tl2, 0, 60/3.6, w2, ml)
    assert Config.V_MIN <= v_r <= w2.v_max, "Tizlik howa çäginden çykdy!"
    print(f"  ✅ GLOSA+ML+Howa maslahat: {v_r*3.6:.1f} km/s ({st})")

    print(f"\n  Ähli {4} synagy geçdi! ✅\n")
    return ml

# ═══════════════════════════════════════════════════════════
#  ESASY BASGANÇAK
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 1. Synaglar + ML öwretme
    ml_predictor = unit_tests_advanced()

    # 2. Demo
    interactive_demo_advanced(ml_predictor)

    # 3. Köpçülikleýin simulýasiýa
    results_glosa, results_no_glosa = run_batch_advanced(
        n=Config.N_SIMS, ml_predictor=ml_predictor)

    # 4. Statistika
    stats = compute_stats_advanced(results_glosa, results_no_glosa)
    print_results_advanced(stats, ml_predictor)

    # 5. Grafik
    plot_results_advanced(stats, results_glosa, results_no_glosa, ml_predictor)

    print("  ✅ Kämil GLOSA simulýasiýasy tamamlandy!\n")
