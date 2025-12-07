import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import os 

# --- INIT PYGAME ---
# Di Linux, buffer terlalu kecil kadang bikin crash, kita naikkan ke 2048 agar aman
pygame.mixer.pre_init(44100, -16, 1, 2048)
pygame.init()

# --- DETEKSI RESOLUSI LAYAR ---
try:
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
except:
    WIDTH, HEIGHT = 1366, 768 # Fallback jika gagal deteksi

CAM_WIDTH, CAM_HEIGHT = 640, 480 

# --- KONFIGURASI WARNA (GOJO STYLE) ---
COLOR_LEFT = (0, 0, 255)      # Biru (Blue/Ao)
COLOR_RIGHT = (255, 0, 0)     # Merah (Red/Aka)
COLOR_FUSION = (143, 0, 255)  # Ungu (Purple/Murasaki)

# --- KONFIGURASI GALAXY ---
NUM_GALAXY_PARTICLES = 350
GALAXY_COLOR_1 = (138, 43, 226)
GALAXY_COLOR_2 = (0, 191, 255)
GALAXY_COLOR_3 = (255, 255, 255)

# --- KONFIGURASI UTAMA LAINNYA ---
BACKGROUND_COLOR = (0, 0, 0)
FOCAL_LENGTH = 400
NUM_PARTICLES = 400       
SPHERE_RADIUS = 150

# --- SETUP SUARA (FIXED) ---
def load_sound(filename, freq=440):
    # Cek apakah file ada
    if os.path.exists(filename):
        try:
            return pygame.mixer.Sound(filename)
        except:
            print(f"Gagal load {filename}, format tidak didukung.")
    
    # --- BAGIAN INI YANG SEBELUMNYA ERROR ---
    # Fallback: Buat bunyi beep sintetis
    print(f"Menggunakan suara beep untuk: {filename}")
    duration = 0.3
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    # PERBAIKAN: 'endpoint=False', bukan 'false=False'
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    sound_array = (wave * 32767).astype(np.int16)
    stereo_array = np.column_stack((sound_array, sound_array))
    return pygame.sndarray.make_sound(stereo_array)


sfx_left   = load_sound("blue.mp3", 440)     # Nada A
sfx_right  = load_sound("red.mp3", 554)   # Nada C#
sfx_fusion = load_sound("purple.mp3", 660)  # Nada E
explode_sfx = load_sound("demeg.mp3", 200)
# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- SETUP LAYAR ---
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Domain Expansion: Gojo Style")
clock = pygame.time.Clock()

fade_surface = pygame.Surface((WIDTH, HEIGHT))
fade_surface.fill(BACKGROUND_COLOR)
fade_surface.set_alpha(40)

# --- FUNGSI UTILITAS ---
def lerp(start, end, t): return start + (end - start) * t
def lerp_color(c1, c2, t):
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return (r, g, b)

# --- CLASS GALAXY ---
class GalaxyParticle:
    def __init__(self):
        self.reset()
        self.angle = np.random.uniform(0, 2 * math.pi)
    def reset(self):
        max_dist = min(WIDTH, HEIGHT) * 0.8
        self.dist = np.random.uniform(50, max_dist) 
        self.angle = np.random.uniform(0, 2 * math.pi)
        self.speed = np.random.uniform(0.02, 0.05)
        self.size = np.random.randint(1, 4)
        choice = random.random()
        if choice < 0.4: self.color = GALAXY_COLOR_1
        elif choice < 0.8: self.color = GALAXY_COLOR_2
        else: self.color = GALAXY_COLOR_3
    def update_and_draw(self, surface, cx, cy, alpha_factor):
        self.angle += self.speed
        x = cx + math.cos(self.angle) * self.dist
        y = cy + math.sin(self.angle) * self.dist * 0.6 
        if alpha_factor > 0.01:
            target_surface = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            r, g, b = self.color
            alpha = int(255 * alpha_factor)
            pygame.draw.circle(target_surface, (r, g, b, alpha), (self.size, self.size), self.size)
            surface.blit(target_surface, (int(x)-self.size, int(y)-self.size))

# --- CLASS BOLA ---
class Particle:
    __slots__ = ('x', 'y', 'z', 'base_x', 'base_y', 'base_z', 'ex_dx', 'ex_dy', 'ex_dz')
    def __init__(self):
        theta = np.random.uniform(0, 2 * math.pi)
        phi = np.random.uniform(0, math.pi)
        ex = np.random.uniform(-1, 1); ey = np.random.uniform(-1, 1); ez = np.random.uniform(-1, 1)
        norm = math.sqrt(ex*ex + ey*ey + ez*ez) or 1
        self.ex_dx = ex/norm; self.ex_dy = ey/norm; self.ex_dz = ez/norm
        r = SPHERE_RADIUS
        self.base_x = r * math.sin(phi) * math.cos(theta)
        self.base_y = r * math.sin(phi) * math.sin(theta)
        self.base_z = r * math.cos(phi)
        self.x, self.y, self.z = self.base_x, self.base_y, self.base_z
    def update_and_draw(self, surface, cx, cy, scale, rot_x, rot_y, expl_val, color):
        c_x, s_x = math.cos(rot_x), math.sin(rot_x)
        c_y, s_y = math.cos(rot_y), math.sin(rot_y)
        y = self.base_y * c_x - self.base_z * s_x
        z = self.base_y * s_x + self.base_z * c_x
        self.base_y, self.base_z = y, z
        x = self.base_x * c_y - self.base_z * s_y
        z = self.base_x * s_y + self.base_z * c_y
        self.base_x, self.base_z = x, z
        
        self.x = self.base_x + (self.ex_dx * expl_val)
        self.y = self.base_y + (self.ex_dy * expl_val)
        self.z = self.base_z + (self.ex_dz * expl_val)
        
        if self.z + FOCAL_LENGTH <= 1: return
        factor = FOCAL_LENGTH / (FOCAL_LENGTH + self.z)
        x_2d = int(self.x * factor * scale + cx)
        y_2d = int(self.y * factor * scale + cy)
        
        if 0 <= x_2d < WIDTH and 0 <= y_2d < HEIGHT:
            size = int(3 * factor * scale)
            if size < 1: size = 1
            if size * scale < 0.1: return
            intensity = factor * 0.8
            if intensity > 1: intensity = 1
            r, g, b = color
            if expl_val > 50: 
                flash = min(255, int((expl_val - 50) * 2))
                r = min(255, r + flash)
                g = min(255, g + flash)
                b = min(255, b + flash)
            else:
                r, g, b = int(r*intensity), int(g*intensity), int(b*intensity)
            pygame.draw.circle(surface, (r, g, b), (x_2d, y_2d), size)

# --- FUNGSI DETEKSI ---
def is_fist(lms):
    wrist_x, wrist_y = lms.landmark[0].x, lms.landmark[0].y
    tip_x, tip_y = lms.landmark[8].x, lms.landmark[8].y
    return ((tip_x - wrist_x)**2 + (tip_y - wrist_y)**2) < 0.04

def is_domain_pose(lms):
    index_up = lms.landmark[8].y < lms.landmark[6].y
    middle_up = lms.landmark[12].y < lms.landmark[10].y
    ring_down = lms.landmark[16].y > lms.landmark[14].y
    pinky_down = lms.landmark[20].y > lms.landmark[18].y
    dist_tips = math.sqrt((lms.landmark[8].x - lms.landmark[12].x)**2 + 
                          (lms.landmark[8].y - lms.landmark[12].y)**2)
    return index_up and middle_up and ring_down and pinky_down and (dist_tips < 0.08)

# --- INIT ---
p_left = [Particle() for _ in range(NUM_PARTICLES)]
p_right = [Particle() for _ in range(NUM_PARTICLES)]
galaxy_particles = [GalaxyParticle() for _ in range(NUM_GALAXY_PARTICLES)]

pos_L = {'x': WIDTH//4, 'y': HEIGHT//2, 's': 1.0, 'fist': False, 'ex': 0.0, 'alpha': 0.0}
pos_R = {'x': 3*WIDTH//4, 'y': HEIGHT//2, 's': 1.0, 'fist': False, 'ex': 0.0, 'alpha': 0.0}

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH); cap.set(4, CAM_HEIGHT); cap.set(5, 60)

running = True
smooth = 0.3

# --- VARIABLES ---
fusion_progress = 0.0
FUSION_SPEED = 0.08
is_fusion_active = False 
color_hold_timer = 0      

domain_active_state = False 
domain_intensity = 0.0       
domain_cooldown = 0  

flag_sound_L = False
flag_sound_R = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    success, img = cap.read()
    det_L, det_R = False, False
    cur_fist_L, cur_fist_R = False, False
    detected_domain_gesture = False 

    if success:
        img = cv2.flip(img, 1)
        img.flags.writeable = False
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.flags.writeable = True

        if results.multi_hand_landmarks:
            for lms, info in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)
                label = info.classification[0].label
                
                cx, cy = int(lms.landmark[9].x * WIDTH), int(lms.landmark[9].y * HEIGHT)
                fist = is_fist(lms)
                
                if is_domain_pose(lms):
                    detected_domain_gesture = True

                x4, y4 = lms.landmark[4].x, lms.landmark[4].y
                x8, y8 = lms.landmark[8].x, lms.landmark[8].y
                scale = max(0.4, ((x4-x8)**2 + (y4-y8)**2) * 10)

                if label == "Left":
                    det_L = True
                    pos_L['x'] += (cx - pos_L['x']) * smooth
                    pos_L['y'] += (cy - pos_L['y']) * smooth
                    pos_L['s'] = scale
                    cur_fist_L = fist
                else:
                    det_R = True
                    pos_R['x'] += (cx - pos_R['x']) * smooth
                    pos_R['y'] += (cy - pos_R['y']) * smooth
                    pos_R['s'] = scale
                    cur_fist_R = fist
        
        cv2.imshow("Cam Monitor", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): running = False

    # --- LOGIC ---
    if detected_domain_gesture and domain_cooldown == 0:
        domain_active_state = not domain_active_state 
        domain_cooldown = 60 
        if explode_sfx: explode_sfx.play() 

    if domain_cooldown > 0: domain_cooldown -= 1
    if domain_active_state: domain_intensity = min(1.0, domain_intensity + 0.05)
    else: domain_intensity = max(0.0, domain_intensity - 0.02)

    # --- ALPHA & SOUND ---
    # KIRI (BIRU)
    target_alpha_L = 0.0
    if det_L:
        if not domain_active_state: target_alpha_L = 1.0
        else:
            if domain_intensity >= 0.95 and cur_fist_L: target_alpha_L = 1.0
            else: target_alpha_L = 0.0

    if pos_L['alpha'] < target_alpha_L: pos_L['alpha'] = min(1.0, pos_L['alpha'] + 0.1)
    else: pos_L['alpha'] = max(0.0, pos_L['alpha'] - 0.1)

    if pos_L['alpha'] > 0.5 and not flag_sound_L and not is_fusion_active:
        if sfx_left: sfx_left.play()
        flag_sound_L = True
    elif pos_L['alpha'] < 0.1:
        flag_sound_L = False 

    # KANAN (MERAH)
    target_alpha_R = 0.0
    if det_R:
        if not domain_active_state: target_alpha_R = 1.0
        else:
            if domain_intensity >= 0.95 and cur_fist_R: target_alpha_R = 1.0
            else: target_alpha_R = 0.0

    if pos_R['alpha'] < target_alpha_R: pos_R['alpha'] = min(1.0, pos_R['alpha'] + 0.1)
    else: pos_R['alpha'] = max(0.0, pos_R['alpha'] - 0.1)

    if pos_R['alpha'] > 0.5 and not flag_sound_R and not is_fusion_active:
        if sfx_right: sfx_right.play()
        flag_sound_R = True
    elif pos_R['alpha'] < 0.1:
        flag_sound_R = False

    # FUSION
    current_fusion_state = (det_L and det_R and cur_fist_L and cur_fist_R)
    
    if current_fusion_state and not is_fusion_active:
        if sfx_fusion: sfx_fusion.play()
    
    if is_fusion_active and not current_fusion_state:
        pos_L['ex'] = 600; pos_R['ex'] = 600
        if explode_sfx: explode_sfx.play()
        color_hold_timer = 60 

    is_fusion_active = current_fusion_state
    target_fusion = 1.0 if current_fusion_state else 0.0
    
    if fusion_progress < target_fusion: fusion_progress = min(target_fusion, fusion_progress + FUSION_SPEED)
    elif fusion_progress > target_fusion: fusion_progress = max(target_fusion, fusion_progress - FUSION_SPEED)

    mid_x = (pos_L['x'] + pos_R['x']) / 2
    mid_y = (pos_L['y'] + pos_R['y']) / 2
    render_x_L = lerp(pos_L['x'], mid_x, fusion_progress)
    render_y_L = lerp(pos_L['y'], mid_y, fusion_progress)
    render_s_L = lerp(pos_L['s'], 1.5, fusion_progress)
    render_x_R = lerp(pos_R['x'], mid_x, fusion_progress)
    render_y_R = lerp(pos_R['y'], mid_y, fusion_progress)
    render_s_R = lerp(pos_R['s'], 1.5, fusion_progress)

    if color_hold_timer > 0:
        curr_color_L = COLOR_FUSION; curr_color_R = COLOR_FUSION
        color_hold_timer -= 1 
    else:
        curr_color_L = lerp_color(COLOR_LEFT, COLOR_FUSION, fusion_progress)
        curr_color_R = lerp_color(COLOR_RIGHT, COLOR_FUSION, fusion_progress)

    rot = 0.02 + (fusion_progress * 0.03)
    
    if color_hold_timer == 0 and fusion_progress < 0.1:
        if pos_L['fist'] and not cur_fist_L and det_L:
            if pos_L['alpha'] > 0.5:
                pos_L['ex'] = 400; 
                if explode_sfx: explode_sfx.play()
        if pos_R['fist'] and not cur_fist_R and det_R:
            if pos_R['alpha'] > 0.5:
                pos_R['ex'] = 400; 
                if explode_sfx: explode_sfx.play()

    pos_L['fist'] = cur_fist_L; pos_R['fist'] = cur_fist_R
    pos_L['ex'] *= 0.85; pos_R['ex'] *= 0.85

    # --- RENDER ---
    screen.blit(fade_surface, (0, 0))

    if domain_intensity > 0:
        for gp in galaxy_particles:
            gp.update_and_draw(screen, WIDTH//2, HEIGHT//2, domain_intensity)

    vis_L = pos_L['alpha']
    vis_R = pos_R['alpha']

    if vis_L > 0.01:
        for p in p_left:
            p.update_and_draw(screen, render_x_L, render_y_L, render_s_L * vis_L, rot, rot*0.5, pos_L['ex'], curr_color_L)
            
    if vis_R > 0.01:
        for p in p_right:
            p.update_and_draw(screen, render_x_R, render_y_R, render_s_R * vis_R, rot*0.5, rot, pos_R['ex'], curr_color_R)

    pygame.display.flip()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()