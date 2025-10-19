# Installation des dépendances 
# !pip install -q opencv-python numpy matplotlib scipy gradio scikit-image

import cv2
import numpy as np
import gradio as gr
import math
import os
from scipy.signal import convolve2d
from skimage.util import random_noise
import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict
import time
import datetime

# --- CONSTANTES GLOBALES ---
AUTHOR = "Hamza Guemmi"  # Identifiant de l'auteur
FAIL_THRESHOLD_EUROS = 0.02  # Seuil de défaillance : erreur de diagnostic >= 2 centimes d'euro.

# Ratios basés sur le diamètre de 2€ (25.75mm)
# Ratios Théoriques (pour référence) : 2€=1.000, 20c=0.864, 2c=0.728
COIN_RATIOS = {
    "2 €": (1.000, 2.00, (255, 165, 0)),
    "20 c": (0.864, 0.20, (255, 255, 0)),
    "2 c": (0.728, 0.02, (0, 0, 255)),
    "Inconnu": (0.00, 0.00, (128, 128, 128))
}
COIN_VALUES = list(COIN_RATIOS.keys())

# Matrice de Quantification Standard (pour compression DCT)
Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

# --- FONCTIONS COMMUNES DE PRÉTRAITEMENT / DÉTECTION ---

def remove_shadows_and_smooth(image, bg_kernel_size=51, gaussian_ksize=5):
    """ Supprime l'illumination / ombres et applique un lissage gaussien. """
    h, w = image.shape[:2]
    bg_kernel_size = max(3, min(h, w, 51))
    if bg_kernel_size % 2 == 0: bg_kernel_size += 1
    if gaussian_ksize % 2 == 0: gaussian_ksize += 1

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel_size, bg_kernel_size))
    background = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
    v_sub = cv2.subtract(v, background)
    v_norm = cv2.normalize(v_sub, None, 0, 255, cv2.NORM_MINMAX)

    if gaussian_ksize > 1:
        v_blur = cv2.GaussianBlur(v_norm, (gaussian_ksize, gaussian_ksize), 0)
    else:
        v_blur = v_norm

    hsv_corr = hsv.copy()
    hsv_corr[:, :, 2] = v_blur.astype(np.uint8)
    corrected_bgr = cv2.cvtColor(hsv_corr, cv2.COLOR_HSV2BGR)
    return corrected_bgr

def enhance_dct_compressed_image(image_bgr, compression_level_estimate=None):
    """
    Améliore les images compressées par DCT en réduisant les artefacts de bloc tout en préservant les bords.
    Le paramètre optionnel compression_level_estimate permet d'ajuster la force du filtrage.
    """
    # 1. Estimation du niveau de compression si non fourni
    if compression_level_estimate is None:
        # Conversion en niveau de gris pour la détection des blocs
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Détection des contours horizontaux et verticaux avec Sobel
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Recherche des limites de bloc potentielles (tous les 8 pixels)
        h, w = gray.shape[:2]
        block_strength_h = 0
        block_strength_v = 0

        # Échantillonner les bords de blocs horizontaux
        for y in range(0, h-1, 8):
            block_strength_h += np.mean(np.abs(sobel_h[y, :]))

        # Échantillonner les bords de blocs verticaux
        for x in range(0, w-1, 8):
            block_strength_v += np.mean(np.abs(sobel_v[:, x]))

        block_strength = (block_strength_h + block_strength_v) / 2
        compression_level_estimate = min(1.0, max(0.0, block_strength / 100))

    # 2. Filtrage bilatéral adaptatif (plus fort pour une compression plus élevée)
    sigma_color = 10 + 20 * compression_level_estimate  # Sigma couleur adaptatif
    sigma_space = 5 + 10 * compression_level_estimate   # Sigma spatial adaptatif

    # Appliquer un filtre bilatéral pour réduire les blocs tout en préservant les bords
    deblocked = cv2.bilateralFilter(image_bgr, 9, sigma_color, sigma_space)

    # 3. Amélioration des bords pour les images compressées
    gray = cv2.cvtColor(deblocked, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 4. Pour une compression plus élevée, appliquer un affûtage supplémentaire adaptatif
    if compression_level_estimate > 0.3:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]]) * (0.5 + compression_level_estimate)

        # Appliquer l'affûtage uniquement aux régions de bord pour éviter d'améliorer les artefacts de bloc
        sharpened = cv2.filter2D(deblocked, -1, kernel)

        # Mélanger les bords affûtés avec les zones lisses déblocées
        alpha = edges_dilated.astype(float) / 255 * min(0.8, compression_level_estimate)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        alpha = alpha[:, :, np.newaxis]  # Ajouter la dimension du canal pour le broadcasting

        # Mélange pondéré
        enhanced = deblocked * (1 - alpha) + sharpened * alpha
        return enhanced.astype(np.uint8)

    return deblocked

def detect_circles_base(image_bgr):
    """ Exécute la détection des cercles et le filtrage des intrus, renvoyant Saturation (s). """
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0: return [], image_bgr.copy()

    image_corrected = remove_shadows_and_smooth(image_bgr, bg_kernel_size=51, gaussian_ksize=5)
    hsv_orig = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(image_corrected, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    processed_gray = cv2.bilateralFilter(clahe.apply(v_channel), 11, 25, 25)
    processed_gray = cv2.medianBlur(processed_gray, 5)

    laplacian_image = cv2.Laplacian(processed_gray, cv2.CV_64F, ksize=3)
    laplacian_image_abs = np.absolute(laplacian_image).astype(np.uint8)

    circles = cv2.HoughCircles(
        processed_gray, cv2.HOUGH_GRADIENT_ALT, dp=1.2, minDist=25, param1=150, param2=0.85, minRadius=35, maxRadius=0
    )

    output = image_corrected.copy()

    # Estimation robuste du fond
    bg_patch_size = 50
    bg_colors = []
    bg_areas = [(0, 0), (0, h - bg_patch_size), (w - bg_patch_size, 0), (w - bg_patch_size, h - bg_patch_size)]
    for x, y in bg_areas:
        if x < w and y < h:
            patch = image_corrected[y:y + bg_patch_size, x:x + bg_patch_size]
            if patch.size > 0:
                bg_colors.append(patch.reshape(-1, 3))

    bg_color = np.median(np.concatenate(bg_colors), axis=0).astype(int) if bg_colors else np.array([128, 128, 128])

    kept_valid_coins = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)
        kept = []

        # Filtrage des Cercles Imbriqués/Redondants
        for circ in sorted_circles:
            x, y, r = circ
            is_redundant = False
            for kx, ky, kr in kept:
                dist = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
                if dist < (kr * 0.15) or (r < kr and dist + r < kr * 1.1):
                    is_redundant = True
                    break
            if not is_redundant:
                kept.append((x, y, r))

        # Filtrage des Intrus (Texture, Variance V, Couleur du Centre)
        for (x, y, r) in kept:
            r_full_area = max(10, int(r * 0.90))
            mask_full_area = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_full_area, (x, y), r_full_area, 255, -1)

            laplacian_values = laplacian_image_abs[mask_full_area == 255]
            mean_laplacian_gradient = np.mean(laplacian_values) if laplacian_values.size > 0 else 0
            TEXTURE_THRESHOLD = 6.0
            is_textured = mean_laplacian_gradient > TEXTURE_THRESHOLD

            v_values = v_channel[mask_full_area == 255]
            v_variance = np.var(v_values) if v_values.size > 0 else 0
            V_VARIANCE_THRESHOLD = 40.0
            is_varied_in_brightness = v_variance > V_VARIANCE_THRESHOLD

            r_color_center = max(5, int(r * 0.30))
            mask_color_center = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask_color_center, (x, y), r_color_center, 255, -1)

            center_pixels_orig = hsv_orig[:, :, :][mask_color_center == 255]

            if center_pixels_orig.size == 0: continue

            # Extraction de la Saturation (s) pour la méthode FIXE
            center_saturation = int(np.mean(center_pixels_orig[:, 1]))

            # Vérification de l'arrière-plan pour les intrus
            center_pixels_corr = image_corrected[mask_color_center == 255]
            center_color_bgr = np.mean(center_pixels_corr, axis=0).astype(np.uint8)
            dist_to_bg = np.linalg.norm(center_color_bgr - bg_color)
            BGR_TOLERANCE = 100
            is_center_like_bg = dist_to_bg < BGR_TOLERANCE

            is_a_valid_coin = is_textured and is_varied_in_brightness
            is_ring_or_smooth_intrus = (not is_textured or not is_varied_in_brightness) and is_center_like_bg

            if not is_a_valid_coin and not is_ring_or_smooth_intrus:
                 is_a_valid_coin = True

            if not is_a_valid_coin or is_ring_or_smooth_intrus:
                continue

            # Stocker (x, y, r, saturation)
            kept_valid_coins.append((x, y, r, center_saturation))

    return kept_valid_coins, output

# --- MÉTHODE 1 : CLASSIFICATION PAR RAYON FIXE (Rigide) ---

def get_coin_value_by_radius_fixed(r, s):
    """ Classifie la pièce selon le rayon (r) et la Saturation (s). (MÉTHODE FIXE) """
    S_SATURATED_MIN = 127

    if r > 150: return "2 €", 2.00
    elif r >= 128 and r <= 150: return "20 c", 0.20
    elif r >= 124 and r < 128:
        if s > S_SATURATED_MIN: return "2 c", 0.02
        else: return "20 c", 0.20
    elif r >= 101 and r < 124: return "2 c", 0.02
    else: return "Inconnu", 0.00

def detect_circles_fixed(image_bgr):
    """ Fonction principale de détection et comptage des pièces (MÉTHODE FIXE). """
    kept_valid_coins, output = detect_circles_base(image_bgr)

    coin_counts = {}
    total_value = 0.0

    for (x, y, r, s) in kept_valid_coins:
        coin_value, value_float = get_coin_value_by_radius_fixed(r, s)

        if coin_value != "Inconnu":
            total_value += value_float
            coin_counts[coin_value] = coin_counts.get(coin_value, 0) + 1

            # Dessin
            coin_color_bgr = COIN_RATIOS[coin_value][2]
            cv2.circle(output, (x, y), r, coin_color_bgr, 4)
            cv2.putText(output, f"{coin_value}", (x - r, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, coin_color_bgr, 2)

    report_lines = [f"{v}: {coin_counts.get(v, 0)} pièces" for v in COIN_VALUES if v != "Inconnu"]
    detailed_report = " | ".join([line for line in report_lines if ": 0 pièces" not in line])
    final_count_text = f"Valeur Totale : {total_value:.2f} € | Classification : {detailed_report}"

    results = {
        'method': 'fixed',
        'total_value': total_value,
        'counts': coin_counts,
        'image': output
    }

    return output, final_count_text, results

# --- MÉTHODE 2 : CLASSIFICATION PAR CALIBRAGE DYNAMIQUE (Robuste) ---

def detect_circles_calibrated(image_bgr):
    """
    Détection et comptage des pièces par calibrage dynamique.
    Classification basée UNIQUEMENT sur le ratio de rayon par rapport à la pièce 2€ (R_ref),
    ignorant la Saturation (s).
    """
    kept_valid_coins, output = detect_circles_base(image_bgr)

    if not kept_valid_coins:
        results = {'method': 'calibrated', 'total_value': 0.0, 'counts': {}, 'image': output}
        return output, "Erreur: Aucune pièce détectée.", results

    # 1. Calibrage: Trouver la pièce de 2€ (la plus grande)
    R_ref = max(r for x, y, r, s in kept_valid_coins)

    # 2. Paramètres de Classification (Centrés sur les Ratios Théoriques)
    R_2C_THEORIQUE = 0.728
    R_20C_THEORIQUE = 0.864
    TOLERANCE_RATIO = 0.05  # Fenêtre de tolérance de 4% autour de la cible théorique

    coin_counts = {}
    total_value = 0.0

    for (x, y, r, s) in kept_valid_coins: # 's' est ignoré dans la classification
        r_ratio = r / R_ref
        coin_value = "Inconnu"
        value_float = 0.0

        # Classification basée UNIQUEMENT sur le ratio de rayon (distance au ratio théorique)

        # 1. 2 Euros
        if r_ratio >= (1.0 - TOLERANCE_RATIO): # Correction: utilisation de >= pour inclure la limite
            coin_value, value_float = "2 €", 2.00

        # 2. 20 Centimes (ratio théorique 0.864)
        elif abs(r_ratio - R_20C_THEORIQUE) < TOLERANCE_RATIO:
             coin_value, value_float = "20 c", 0.20

        # 3. 2 Centimes (ratio théorique 0.728)
        elif abs(r_ratio - R_2C_THEORIQUE) < TOLERANCE_RATIO:
             coin_value, value_float = "2 c", 0.02

        # 4. Traitement des cas ambigus (entre les fenêtres de tolérance)
        elif r_ratio > R_2C_THEORIQUE - TOLERANCE_RATIO and r_ratio < 1.0 - TOLERANCE_RATIO / 2:
            # Si le ratio tombe entre les fenêtres, on le classe par la cible théorique la plus proche.
            diff_20c = abs(r_ratio - R_20C_THEORIQUE)
            diff_2c = abs(r_ratio - R_2C_THEORIQUE)

            if diff_20c < diff_2c:
                 coin_value, value_float = "20 c", 0.20
            else:
                 coin_value, value_float = "2 c", 0.02

        # 5. Pièce trop petite ou trop éloignée
        else:
             coin_value, value_float = "Inconnu", 0.00

        if coin_value != "Inconnu":
            total_value += value_float
            coin_counts[coin_value] = coin_counts.get(coin_value, 0) + 1

            # Dessin
            coin_color_bgr = COIN_RATIOS[coin_value][2]
            cv2.circle(output, (x, y), r, coin_color_bgr, 4)
            # Affiche le ratio pour le débogage
            cv2.putText(output, f"{coin_value} R={r_ratio:.3f}", (x - r, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, coin_color_bgr, 2)

    report_lines = [f"{v}: {coin_counts.get(v, 0)} pièces" for v in COIN_VALUES if v != "Inconnu"]
    detailed_report = " | ".join([line for line in report_lines if ": 0 pièces" not in line])
    final_count_text = f"Valeur Totale : {total_value:.2f} € | Classification : {detailed_report} (R_ref={R_ref:.2f})"

    results = {
        'method': 'calibrated',
        'total_value': total_value,
        'counts': coin_counts,
        'image': output
    }

    return output, final_count_text, results


# --- FONCTIONS UTILES DE MÉTRIQUES ---

def calculate_Q(I_original, I_bruited):
    """ Calcule l'écart Q sur le canal V (luminosité). """
    I_orig_v = cv2.cvtColor(I_original, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
    I_bruited_v = cv2.cvtColor(I_bruited, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
    mse = np.mean((I_orig_v - I_bruited_v) ** 2)
    Q = np.sqrt(mse)
    return Q

def calculate_PSNR(I_original, I_compressed):
    """ Calcule le PSNR sur le canal V. """
    I_orig_v = cv2.cvtColor(I_original, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
    I_comp_v = cv2.cvtColor(I_compressed, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float64)
    mse = np.mean((I_orig_v - I_comp_v) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_entropy(image_data):
    """ Calcule l'entropie d'un signal 8-bit. """
    if image_data.size == 0: return 0.0
    hist = np.histogram(image_data.flatten(), bins=256, range=(0, 256))[0]
    probabilities = hist / hist.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def generate_plot_b64(x_data, y_data, title, xlabel, ylabel, fail_x=None, fail_y=None, secondary_fail_x=None, secondary_fail_y=None, secondary_label=""):
    """ Génère un graphique Matplotlib et le renvoie en base64 pour Gradio. """
    plt.figure(figsize=(8, 5))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label=f'{ylabel} vs. {xlabel}')

    # Premier point de défaillance (généralement Fixe)
    if fail_x is not None and fail_y is not None:
        plt.scatter(fail_x, fail_y, color='red', marker='X', s=200, label=f'Défaillance (Fixe)', zorder=5)

    # Deuxième point de défaillance (généralement Calibrée)
    if secondary_fail_x is not None and secondary_fail_y is not None:
        plt.scatter(secondary_fail_x, secondary_fail_y, color='green', marker='P', s=200, label=f'Défaillance (Calibrée)', zorder=5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if fail_x is not None or secondary_fail_x is not None:
        plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_b64}" />'

# Fonction utilitaire pour encoder une image en base64
def encode_img_b64(img_bgr):
    success, img_b64_arr = cv2.imencode('.png', img_bgr)
    if success:
        return base64.b64encode(img_b64_arr.tobytes()).decode('utf-8')
    return ""

# --- PROBLÈME 2 : BRUIT SEL ET POIVRE  ---

def apply_filters_and_report(img_bgr, orig_value, method_func):
    """ Applique des filtres et trouve le meilleur réparateur. """
    filters = {
        "Médian (k=5)": lambda img: cv2.medianBlur(img, 5),
        "Gaussien (k=5)": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
        "Bilatéral": lambda img: cv2.bilateralFilter(img, 9, 75, 75)
    }

    best_filter = {"name": "Aucun", "error": FAIL_THRESHOLD_EUROS * 10, "value": 0.0, "image_bgr": None}
    filter_results = []

    for filter_name, filter_func in filters.items():
        filtered_img_bgr = filter_func(img_bgr)
        # Exécuter la méthode de classification spécifiée (fixed ou calibrated)
        _, _, filtered_res = method_func(filtered_img_bgr)
        filtered_value = filtered_res['total_value']
        filtered_error = abs(filtered_value - orig_value)

        error_status = "✅ Réparé" if filtered_error < FAIL_THRESHOLD_EUROS else "❌ Échec"

        filter_results.append({
            'name': filter_name,
            'value': filtered_value,
            'error': filtered_error,
            'status': error_status,
            'image_bgr': filtered_res['image']
        })

        if filtered_error < best_filter['error']:
            best_filter = {"name": filter_name, "value": filtered_value, "error": filtered_error, "image_bgr": filtered_res['image']}

    # Si le meilleur filtre ne répare pas, le résultat est l'échec.
    if best_filter['error'] >= FAIL_THRESHOLD_EUROS:
         best_filter = {"name": "Aucun (Échec de la réparation)", "error": best_filter['error'], "value": best_filter['value'], "image_bgr": best_filter['image_bgr']} # Conserver l'image (même si elle est défaillante)

    return filter_results, best_filter

def study_noise(original_image_data, original_results_list_fixed, original_results_list_calibrated):
    """ Réalise l'étude complète du bruit Sel et Poivre pour les deux méthodes, incluant les filtres. """
    if not original_image_data or not original_results_list_fixed or not original_results_list_calibrated:
        return "Veuillez d'abord exécuter les Sections 1 et 2 pour obtenir les résultats de base.", "", {}, {}

    results_html = []

    for i, (orig_img_bgr, original_res_fixed, original_res_calibrated) in enumerate(zip(original_image_data, original_results_list_fixed, original_results_list_calibrated)):

        orig_img_bgr = orig_img_bgr.astype(np.uint8)
        image_name = original_res_fixed['name']
        orig_value = original_res_fixed['total_value']

        d_values = np.linspace(0.0, 0.25, 20)

        study_data = {
            'fixed': {'Q_values': [], 'fail_d': None, 'fail_Q': None, 'fail_img_bgr': None, 'fail_bruited_img_bgr': None, 'results': []},
            'calibrated': {'Q_values': [], 'fail_d': None, 'fail_Q': None, 'fail_img_bgr': None, 'fail_bruited_img_bgr': None, 'results': []}
        }

        # Balayage des densités de bruit
        for d in d_values:
            if d == 0:
                bruited_img = orig_img_bgr.copy()
                Q = 0.0
            else:
                temp_img = orig_img_bgr.astype(np.float64) / 255.0
                bruited_float = random_noise(temp_img, mode='s&p', amount=d)
                bruited_img = (bruited_float * 255).astype(np.uint8)
                Q = calculate_Q(orig_img_bgr, bruited_img)

            # La pièce est comptée sur l'image bruitée, incluant le prétraitement à l'intérieur.
            _, _, noisy_res_fixed = detect_circles_fixed(bruited_img)
            error_fixed = abs(noisy_res_fixed['total_value'] - orig_value)

            _, _, noisy_res_calibrated = detect_circles_calibrated(bruited_img)
            error_calibrated = abs(noisy_res_calibrated['total_value'] - orig_value)

            study_data['fixed']['Q_values'].append(Q)
            study_data['calibrated']['Q_values'].append(Q)

            # Détection de la défaillance (FIXE)
            if study_data['fixed']['fail_d'] is None and error_fixed >= FAIL_THRESHOLD_EUROS:
                study_data['fixed']['fail_d'] = d
                study_data['fixed']['fail_Q'] = Q
                study_data['fixed']['fail_img_bgr'] = noisy_res_fixed['image']
                study_data['fixed']['fail_bruited_img_bgr'] = bruited_img.copy()

            # Détection de la défaillance (CALIBRÉE)
            if study_data['calibrated']['fail_d'] is None and error_calibrated >= FAIL_THRESHOLD_EUROS:
                study_data['calibrated']['fail_d'] = d
                study_data['calibrated']['fail_Q'] = Q
                study_data['calibrated']['fail_img_bgr'] = noisy_res_calibrated['image']
                study_data['calibrated']['fail_bruited_img_bgr'] = bruited_img.copy()

            study_data['fixed']['results'].append({'d': d, 'Q': Q, 'value': noisy_res_fixed['total_value'], 'error': error_fixed})
            study_data['calibrated']['results'].append({'d': d, 'Q': Q, 'value': noisy_res_calibrated['total_value'], 'error': error_calibrated})


        # --- 1. Graphique Q(d) ---

        graph_Q_d = generate_plot_b64(
            d_values, study_data['fixed']['Q_values'],
            f'Robustesse au Bruit (Q vs. d) - {image_name}',
            'Densité de Bruit d', 'Écart Q',
            fail_x=study_data['fixed']['fail_d'],
            fail_y=study_data['fixed']['fail_Q'],
            secondary_fail_x=study_data['calibrated']['fail_d'],
            secondary_fail_y=study_data['calibrated']['fail_Q']
        )

        # --- 2. Comparaison des Seuils ---
        fail_d_fixed = study_data['fixed']['fail_d']
        fail_Q_fixed = study_data['fixed']['fail_Q']
        fail_d_calibrated = study_data['calibrated']['fail_d']
        fail_Q_calibrated = study_data['calibrated']['fail_Q']

        fail_Q_fixed_str = f"{fail_Q_fixed:.2f}" if fail_Q_fixed is not None else "N/A"
        fail_d_fixed_str = f"{fail_d_fixed:.3f}" if fail_d_fixed is not None else "N/A"

        fail_Q_cal_str = f"{fail_Q_calibrated:.2f}" if fail_Q_calibrated is not None else "N/A"
        fail_d_cal_str = f"{fail_d_calibrated:.3f}" if fail_d_calibrated is not None else "N/A"

        comparison_html = f"""
        <h4>Synthèse des Seuils de Défaillance (Bruit S&P)</h4>
        <table style='width:100%'>
            <tr>
                <th>Méthode</th>
                <th>$d$ de Défaillance</th>
                <th>$Q$ de Défaillance</th>
                <th>Diagnostic plus Robuste ?</th>
            </tr>
            <tr>
                <td>**Contraintes Fixes**</td>
                <td style="color: {'red' if (fail_d_fixed is not None and fail_d_calibrated is not None and fail_d_fixed < fail_d_calibrated) else 'inherit'}">**{fail_d_fixed_str}**</td>
                <td style="color: {'red' if (fail_Q_fixed is not None and fail_Q_calibrated is not None and fail_Q_fixed < fail_Q_calibrated) else 'inherit'}">**{fail_Q_fixed_str}**</td>
                <td>{'Non' if (fail_d_fixed is not None and fail_d_calibrated is not None and fail_d_fixed < fail_d_calibrated) else 'Oui' if fail_d_calibrated is not None and fail_d_fixed is not None else 'N/A'}</td>
            </tr>
            <tr>
                <td>**Calibrage Dynamique**</td>
                <td style="color: {'green' if (fail_d_calibrated is not None and fail_d_fixed is not None and fail_d_calibrated > fail_d_fixed) else 'inherit'}">**{fail_d_cal_str}**</td>
                <td style="color: {'green' if (fail_Q_calibrated is not None and fail_Q_fixed is not None and fail_Q_calibrated > fail_Q_fixed) else 'inherit'}">**{fail_Q_cal_str}**</td>
                <td>{'Oui' if (fail_d_calibrated is not None and fail_d_fixed is not None and fail_d_calibrated > fail_d_fixed) else 'Non' if fail_d_fixed is not None and fail_d_calibrated is not None else 'N/A'}</td>
            </tr>
        </table>
        """

        # --- 3. Analyse des Filtres Correcteurs ---
        filter_analysis_html = "<h4>Analyse des Filtres Correcteurs ($d$ de Défaillance)</h4>"

        # Méthode Fixe
        best_filter_fixed = {'name': 'N/A', 'value': 0.0, 'error': FAIL_THRESHOLD_EUROS * 10, 'image_bgr': None}
        if study_data['fixed']['fail_bruited_img_bgr'] is not None:
             filter_results_fixed, best_filter_fixed = apply_filters_and_report(study_data['fixed']['fail_bruited_img_bgr'], orig_value, detect_circles_fixed)

             filter_analysis_html += f"<h5>Méthode FIXE ($d={fail_d_fixed_str}$) : Meilleur Filtre : **{best_filter_fixed['name']}** (Erreur: {best_filter_fixed['error']:.2f} €)</h5>"
             filter_analysis_html += "<table style='width:100%'><tr><th>Filtre</th><th>Diagnostic (€)</th><th>Erreur (€)</th><th>Statut</th></tr>"
             for res in filter_results_fixed:
                 row_style = "background-color:#dfd;" if res['status'] == "✅ Réparé" else ""
                 filter_analysis_html += f"<tr style='{row_style}'><td>{res['name']}</td><td>{res['value']:.2f}</td><td>{res['error']:.2f}</td><td>{res['status']}</td></tr>"
             filter_analysis_html += "</table>"

        # Méthode Calibrée
        best_filter_cal = {'name': 'N/A', 'value': 0.0, 'error': FAIL_THRESHOLD_EUROS * 10, 'image_bgr': None}
        if study_data['calibrated']['fail_bruited_img_bgr'] is not None:
             filter_results_cal, best_filter_cal = apply_filters_and_report(study_data['calibrated']['fail_bruited_img_bgr'], orig_value, detect_circles_calibrated)

             filter_analysis_html += f"<h5>Méthode CALIBRÉE ($d={fail_d_cal_str}$) : Meilleur Filtre : **{best_filter_cal['name']}** (Erreur: {best_filter_cal['error']:.2f} €)</h5>"
             filter_analysis_html += "<table style='width:100%'><tr><th>Filtre</th><th>Diagnostic (€)</th><th>Erreur (€)</th><th>Statut</th></tr>"
             for res in filter_results_cal:
                 row_style = "background-color:#dfd;" if res['status'] == "✅ Réparé" else ""
                 filter_analysis_html += f"<tr style='{row_style}'><td>{res['name']}</td><td>{res['value']:.2f}</td><td>{res['error']:.2f}</td><td>{res['status']}</td></tr>"
             filter_analysis_html += "</table>"


        # --- 4. Affichage des Images Clés ---
        images_html = "<h4>Images Clés (Défaillance et Correction)</h4>"

        # Images Fixe
        img_fixed_fail_b64 = encode_img_b64(study_data['fixed']['fail_img_bgr']) if study_data['fixed']['fail_img_bgr'] is not None else ""
        img_fixed_corr_b64 = encode_img_b64(best_filter_fixed['image_bgr']) if best_filter_fixed['image_bgr'] is not None else ""

        # Assurez-vous que l'indice [-1] existe avant d'y accéder, sinon utilisez une valeur par défaut
        fixed_value_at_fail = study_data['fixed']['results'][-1]['value'] if study_data['fixed']['results'] else 0.0

        images_html += f"""
        <div style="display: flex; justify-content: space-around; margin-top: 20px;">
            <div style="width: 48%; text-align: center;">
                <p>Défaillance Fixe ($d={fail_d_fixed_str}$) : {original_res_fixed['total_value']:.2f} € &rarr; {fixed_value_at_fail:.2f} €</p>
                <img src="data:image/png;base64,{img_fixed_fail_b64}" style="max-width: 100%; border: 2px solid red;">
            </div>
            <div style="width: 48%; text-align: center;">
                <p>Correction Fixe ({best_filter_fixed['name']}) : {best_filter_fixed['value']:.2f} €</p>
                <img src="data:image/png;base64,{img_fixed_corr_b64}" style="max-width: 100%; border: 2px solid green;">
            </div>
        </div>
        """

        # Images Calibrée
        img_cal_fail_b64 = encode_img_b64(study_data['calibrated']['fail_img_bgr']) if study_data['calibrated']['fail_img_bgr'] is not None else ""
        img_cal_corr_b64 = encode_img_b64(best_filter_cal['image_bgr']) if best_filter_cal['image_bgr'] is not None else ""

        calibrated_value_at_fail = study_data['calibrated']['results'][-1]['value'] if study_data['calibrated']['results'] else 0.0

        images_html += f"""
        <div style="display: flex; justify-content: space-around; margin-top: 20px;">
            <div style="width: 48%; text-align: center;">
                <p>Défaillance Calibrée ($d={fail_d_cal_str}$) : {original_res_calibrated['total_value']:.2f} € &rarr; {calibrated_value_at_fail:.2f} €</p>
                <img src="data:image/png;base64,{img_cal_fail_b64}" style="max-width: 100%; border: 2px solid red;">
            </div>
            <div style="width: 48%; text-align: center;">
                <p>Correction Calibrée ({best_filter_cal['name']}) : {best_filter_cal['value']:.2f} €</p>
                <img src="data:image/png;base64,{img_cal_corr_b64}" style="max-width: 100%; border: 2px solid green;">
            </div>
        </div>
        """

        # --- Assemblage final ---
        results_html.append(f"""
            <div style="border: 2px solid #ccc; padding: 15px; margin-bottom: 20px;">
                <h2>Problème 2 : Étude du Bruit Sel et Poivre - {image_name}</h2>
                {graph_Q_d}
                {comparison_html}
                {filter_analysis_html}
                {images_html}
            </div>
        """)

    fail_d_calibrated_final = study_data['calibrated']['fail_d']
    fail_Q_calibrated_final = study_data['calibrated']['fail_Q']

    return "Étude du bruit terminée. Comparaison des deux méthodes et analyse des filtres effectuée.", "".join(results_html), fail_d_calibrated_final, fail_Q_calibrated_final

# --- PROBLÈME 3 : COMPRESSION DCT (Analyse Détaillée) ---

def compress_decompress_dct(image_bgr, Q_factor):
    """ Compresse/Décompresse une image en utilisant la DCT avec quantification par Q_factor. """
    h, w = image_bgr.shape[:2]
    ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y_channel = ycbcr[:, :, 0].astype(np.float32)
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded_Y = cv2.copyMakeBorder(Y_channel, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    H, W = padded_Y.shape
    quantization_matrix = Q_MATRIX * Q_factor
    quantization_matrix[quantization_matrix == 0] = 1

    quantized_coefficients = []
    decompressed_Y = np.zeros((H, W), dtype=np.float32)
    orig_entropy = calculate_entropy(Y_channel[:h, :w].astype(np.uint8))

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = cv2.dct(padded_Y[i:i+8, j:j+8] - 128.0)
            quantized_block = np.round(block / quantization_matrix).astype(np.int16)
            quantized_coefficients.extend(quantized_block.flatten())
            dequantized_block = quantized_block * quantization_matrix
            decompressed_Y[i:i+8, j:j+8] = cv2.idct(dequantized_block) + 128.0

    decompressed_Y = np.clip(decompressed_Y, 0, 255).astype(np.uint8)[:h, :w]
    ycbcr_decompressed = ycbcr.copy()
    ycbcr_decompressed[:, :, 0] = decompressed_Y
    decompressed_bgr = cv2.cvtColor(ycbcr_decompressed, cv2.COLOR_YCrCb2BGR)

    coeffs_array = np.array(quantized_coefficients)
    entropy_coeff = calculate_entropy(coeffs_array + 128)
    bpp_compressed = entropy_coeff / 8
    TC = orig_entropy / bpp_compressed if bpp_compressed > 0 else float('inf')

    return decompressed_bgr, TC, orig_entropy, entropy_coeff

def generate_method_report_compression(study_res, image_name, orig_value, method_name, detect_func):

    Q_factors = [res['Q_factor'] for res in study_res['results']]
    PSNR_values = [res['PSNR'] for res in study_res['results']]
    TC_values = [res['TC'] for res in study_res['results']]

    fail_Q_factor = study_res['fail_Q_factor']
    fail_PSNR = study_res['fail_PSNR']
    fail_TC = study_res['fail_TC']

    fail_Q_str = f"{fail_Q_factor:.2f}" if fail_Q_factor is not None else "N/A"
    fail_TC_str = f"{fail_TC:.2f}" if fail_TC is not None else "N/A"
    fail_PSNR_str = f"{fail_PSNR:.2f} dB" if fail_PSNR is not None else "N/A"

    # Graphe PSNR(TC)
    graph_PSNR_TC = generate_plot_b64(
        TC_values, PSNR_values,
        f'Robustesse à la Compression (PSNR vs. TC) - {image_name} ({method_name})',
        'Taux de Compression (TC)', 'PSNR (dB)',
        fail_x=fail_TC, fail_y=fail_PSNR
    )

    # Tableau de balayage
    table_html_Q = "<h4>Résultats du Balayage (Facteur Q)</h4>"
    table_html_Q += "<table style='width:100%'><tr><th>Facteur Q</th><th>TC</th><th>PSNR (dB)</th><th>Diagnostic (€)</th><th>Erreur (€)</th><th>Statut</th></tr>"
    for res in study_res['results']:
        error_status = "❌ Échec" if res['error'] >= FAIL_THRESHOLD_EUROS else "✅ Correct"
        row_style = "background-color:#fdd;" if res['error'] >= FAIL_THRESHOLD_EUROS else ""
        table_html_Q += f"<tr style='{row_style}'><td>{res['Q_factor']:.2f}</td><td>{res['TC']:.2f}</td><td>{res['PSNR']:.2f}</td><td>{res['value']:.2f}</td><td>{res['error']:.2f}</td><td>{error_status}</td></tr>"
    table_html_Q += "</table>"

    # Image de Défaillance
    diag_fail = ""
    if study_res['fail_img_bgr'] is not None:
         img_fail_b64 = encode_img_b64(study_res['fail_img_bgr'])
         diag_fail = f"""
         <h4>Image à la Défaillance (Q={fail_Q_str}, TC={fail_TC_str})</h4>
         <img src="data:image/png;base64,{img_fail_b64}" alt="Image compressée à la défaillance" style="max-width:400px; display: block; margin: 10px auto;">
         """

    # Résumé
    summary = f"""
    <div style="border: 1px dashed #bbb; padding: 10px; margin-top: 15px;">
        <h3>Méthode : {method_name}</h3>
        <p>Seuil de Défaillance (Erreur $\ge$ {FAIL_THRESHOLD_EUROS:.2f} €) :</p>
        <ul>
            <li>**Facteur Q** : **{fail_Q_str}**</li>
            <li>**Taux de Compression (TC)** : **{fail_TC_str}**</li>
            <li>**PSNR** : **{fail_PSNR_str}**</li>
        </ul>
        {graph_PSNR_TC}
        {table_html_Q}
        {diag_fail}
    </div>
    """
    return summary

def study_compression(original_image_data, original_results_list_fixed, original_results_list_calibrated):
    """ Réalise l'étude complète de la compression DCT pour les deux méthodes. """
    if not original_image_data or not original_results_list_fixed or not original_results_list_calibrated:
        return "Veuillez d'abord exécuter les Sections 1 et 2 pour obtenir les résultats de base.", "", {}, {}, None

    results_html = []
    all_study_data = []  # Pour stocker les données d'étude pour toutes les images

    for i, (orig_img_bgr, original_res_fixed, original_res_calibrated) in enumerate(zip(original_image_data, original_results_list_fixed, original_results_list_calibrated)):

        orig_img_bgr = orig_img_bgr.astype(np.uint8)
        image_name = original_res_fixed['name']
        orig_value = original_res_fixed['total_value']

        Q_factors = np.linspace(0.5, 10.0, 20)

        study_data = {
            'fixed': {'PSNR_values': [], 'TC_values': [], 'fail_PSNR': None, 'fail_TC': None, 'fail_Q_factor': None, 'fail_img_bgr': None, 'results': []},
            'calibrated': {'PSNR_values': [], 'TC_values': [], 'fail_PSNR': None, 'fail_TC': None, 'fail_Q_factor': None, 'fail_img_bgr': None, 'results': []}
        }

        # Stockage des données de compression pour chaque Q_factor
        compression_data = []

        # Balayage des facteurs de quantification
        for Q_factor in Q_factors:
            compressed_img, TC, _, _ = compress_decompress_dct(orig_img_bgr, Q_factor)
            PSNR = calculate_PSNR(orig_img_bgr, compressed_img)

            # Stocker les données de compression pour réutilisation
            compression_data.append({
                'Q_factor': Q_factor,
                'compressed_img': compressed_img.copy(),
                'TC': TC,
                'PSNR': PSNR
            })

            # Analyse pour la méthode FIXE
            _, _, comp_res_fixed = detect_circles_fixed(compressed_img)
            error_fixed = abs(comp_res_fixed['total_value'] - orig_value)

            # Analyse pour la méthode CALIBRÉE
            _, _, comp_res_calibrated = detect_circles_calibrated(compressed_img)
            error_calibrated = abs(comp_res_calibrated['total_value'] - orig_value)

            # Stockage des métriques
            study_data['fixed']['PSNR_values'].append(PSNR)
            study_data['fixed']['TC_values'].append(TC)
            study_data['calibrated']['PSNR_values'].append(PSNR)
            study_data['calibrated']['TC_values'].append(TC)

            # Détection de la défaillance (FIXE)
            if study_data['fixed']['fail_Q_factor'] is None and error_fixed >= FAIL_THRESHOLD_EUROS:
                study_data['fixed']['fail_Q_factor'] = Q_factor
                study_data['fixed']['fail_PSNR'] = PSNR
                study_data['fixed']['fail_TC'] = TC
                study_data['fixed']['fail_img_bgr'] = comp_res_fixed['image']

            study_data['fixed']['results'].append({'Q_factor': Q_factor, 'PSNR': PSNR, 'TC': TC, 'value': comp_res_fixed['total_value'], 'error': error_fixed})

            # Détection de la défaillance (CALIBRÉE)
            if study_data['calibrated']['fail_Q_factor'] is None and error_calibrated >= FAIL_THRESHOLD_EUROS:
                study_data['calibrated']['fail_Q_factor'] = Q_factor
                study_data['calibrated']['fail_PSNR'] = PSNR
                study_data['calibrated']['fail_TC'] = TC
                study_data['calibrated']['fail_img_bgr'] = comp_res_calibrated['image']

            study_data['calibrated']['results'].append({'Q_factor': Q_factor, 'PSNR': PSNR, 'TC': TC, 'value': comp_res_calibrated['total_value'], 'error': error_calibrated})


        # --- 1. Rapports Détaillés par Méthode ---

        report_fixed = generate_method_report_compression(study_data['fixed'], image_name, orig_value, "Contraintes Fixes", detect_circles_fixed)
        report_calibrated = generate_method_report_compression(study_data['calibrated'], image_name, orig_value, "Calibrage Dynamique", detect_circles_calibrated)

        # --- 2. Tableau de comparaison des seuils ---
        fail_PSNR_fixed = study_data['fixed']['fail_PSNR']
        fail_TC_fixed = study_data['fixed']['fail_TC']
        fail_PSNR_calibrated = study_data['calibrated']['fail_PSNR']
        fail_TC_calibrated = study_data['calibrated']['fail_TC']

        fail_PSNR_fixed_str = f"{fail_PSNR_fixed:.2f} dB" if fail_PSNR_fixed is not None else "N/A"
        fail_TC_fixed_str = f"{fail_TC_fixed:.2f}" if fail_TC_fixed is not None else "N/A"

        fail_PSNR_cal_str = f"{fail_PSNR_calibrated:.2f} dB" if fail_PSNR_calibrated is not None else "N/A"
        fail_TC_cal_str = f"{fail_TC_calibrated:.2f}" if fail_TC_calibrated is not None else "N/A"

        comparison_html = f"""
        <h4>Synthèse des Seuils de Défaillance (Compression DCT)</h4>
        <table style='width:100%'>
            <tr>
                <th>Méthode</th>
                <th>TC de Défaillance</th>
                <th>PSNR de Défaillance</th>
                <th>Diagnostic plus Robuste ?</th>
            </tr>
            <tr>
                <td>**Contraintes Fixes**</td>
                <td style="color: {'red' if (fail_TC_fixed is not None and fail_TC_calibrated is not None and fail_TC_fixed < fail_TC_calibrated) else 'inherit'}">**{fail_TC_fixed_str}**</td>
                <td style="color: {'red' if (fail_PSNR_fixed is not None and fail_PSNR_calibrated is not None and fail_PSNR_fixed > fail_PSNR_calibrated) else 'inherit'}">**{fail_PSNR_fixed_str}**</td>
                <td>{'Non' if (fail_TC_fixed is not None and fail_TC_calibrated is not None and fail_TC_fixed < fail_TC_calibrated) else 'Oui' if fail_TC_calibrated is not None and fail_TC_fixed is not None else 'N/A'}</td>
            </tr>
            <tr>
                <td>**Calibrage Dynamique**</td>
                <td style="color: {'green' if (fail_TC_calibrated is not None and fail_TC_fixed is not None and fail_TC_calibrated > fail_TC_fixed) else 'inherit'}">**{fail_TC_cal_str}**</td>
                <td style="color: {'green' if (fail_PSNR_calibrated is not None and fail_PSNR_fixed is not None and fail_PSNR_calibrated < fail_PSNR_fixed) else 'inherit'}">**{fail_PSNR_cal_str}**</td>
                <td>{'Oui' if (fail_TC_calibrated is not None and fail_TC_fixed is not None and fail_TC_calibrated > fail_TC_fixed) else 'Non' if fail_TC_fixed is not None and fail_TC_calibrated is not None else 'N/A'}</td>
            </tr>
        </table>
        <p style="font-style:italic;">Une valeur **TC** plus élevée ou **PSNR** plus faible indique une plus grande robustesse (la défaillance survient pour une dégradation plus sévère).</p>
        """

        # --- Assemblage du HTML pour l'image ---
        results_html.append(f"""
            <div style="border: 2px solid #ccc; padding: 15px; margin-bottom: 20px;">
                <h2>Problème 3 : Étude de la Compression DCT - {image_name}</h2>
                {comparison_html}
                <div style="display: flex; justify-content: space-around;">
                    <div style="width: 48%;"> {report_fixed} </div>
                    <div style="width: 48%;"> {report_calibrated} </div>
                </div>
            </div>
        """)

        # Ajouter les données d'étude et de compression pour cette image
        all_study_data.append({
            'image_name': image_name,
            'orig_value': orig_value,
            'orig_img_bgr': orig_img_bgr,
            'study_data': study_data,
            'compression_data': compression_data
        })

    # Retourne les résultats, les métriques et les données d'étude pour réutilisation
    fail_PSNR_calibrated_final = study_data['calibrated']['fail_PSNR']
    fail_TC_calibrated_final = study_data['calibrated']['fail_TC']

    return "Étude de la compression terminée. Comparaison des deux méthodes effectuée.", "".join(results_html), fail_PSNR_calibrated_final, fail_TC_calibrated_final, all_study_data

# --- PROBLÈME 3 COMPRESSION DCT AVEC TRAITEMENT ANTI-ARTIFACTS ---

def study_compression_enhanced(original_image_data, original_results_list_fixed, original_results_list_calibrated, standard_study_data=None):
    """
    Réalise l'étude complète de la compression DCT AVEC TRAITEMENT ANTI-ARTIFACTS.
    Réutilise les données du problème 3 standard pour garantir la cohérence des résultats.
    """
    if not original_image_data or not original_results_list_fixed or not original_results_list_calibrated:
        return "Veuillez d'abord exécuter les Sections 1 et 2 pour obtenir les résultats de base.", "", {}, {}

    if standard_study_data is None:
        return "Veuillez d'abord exécuter le Problème 3 standard pour obtenir les données de référence.", "", {}, {}

    results_html = []

    for i, study_data_item in enumerate(standard_study_data):
        orig_img_bgr = study_data_item['orig_img_bgr']
        image_name = study_data_item['image_name']
        orig_value = study_data_item['orig_value']
        compression_data = study_data_item['compression_data']
        standard_study_data_results = study_data_item['study_data']

        # Récupérer les résultats standards pour cette image
        fixed_standard = standard_study_data_results['fixed']
        calibrated_standard = standard_study_data_results['calibrated']

        # Créer les structures pour les résultats améliorés
        enhanced_study_data = {
            'fixed_enhanced': {'PSNR_values': [], 'TC_values': [], 'fail_PSNR': None, 'fail_TC': None, 'fail_Q_factor': None, 'fail_img_bgr': None, 'results': []},
            'calibrated_enhanced': {'PSNR_values': [], 'TC_values': [], 'fail_PSNR': None, 'fail_TC': None, 'fail_Q_factor': None, 'fail_img_bgr': None, 'results': []}
        }

        # Utiliser les mêmes données de compression que l'étude standard
        for compression_item in compression_data:
            Q_factor = compression_item['Q_factor']
            compressed_img = compression_item['compressed_img']
            TC = compression_item['TC']
            PSNR = compression_item['PSNR']

            # Calcul de l'estimation du niveau de compression (pour l'amélioration adaptative)
            compression_level = min(1.0, Q_factor / 10.0)

            # Version améliorée avec réduction des artefacts DCT
            enhanced_img = enhance_dct_compressed_image(compressed_img, compression_level_estimate=compression_level)

            # --- MÉTHODES AMÉLIORÉES ---

            # Analyse pour la méthode FIXE (améliorée)
            _, _, comp_res_fixed_enhanced = detect_circles_fixed(enhanced_img)
            error_fixed_enhanced = abs(comp_res_fixed_enhanced['total_value'] - orig_value)

            # Analyse pour la méthode CALIBRÉE (améliorée)
            _, _, comp_res_calibrated_enhanced = detect_circles_calibrated(enhanced_img)
            error_calibrated_enhanced = abs(comp_res_calibrated_enhanced['total_value'] - orig_value)

            # Stockage des métriques
            for method in enhanced_study_data:
                enhanced_study_data[method]['PSNR_values'].append(PSNR)
                enhanced_study_data[method]['TC_values'].append(TC)

            # Stockage des résultats enhanced
            enhanced_study_data['fixed_enhanced']['results'].append({
                'Q_factor': Q_factor, 'PSNR': PSNR, 'TC': TC,
                'value': comp_res_fixed_enhanced['total_value'],
                'error': error_fixed_enhanced
            })

            enhanced_study_data['calibrated_enhanced']['results'].append({
                'Q_factor': Q_factor, 'PSNR': PSNR, 'TC': TC,
                'value': comp_res_calibrated_enhanced['total_value'],
                'error': error_calibrated_enhanced
            })

            # Détection de la défaillance (FIXE enhanced)
            if enhanced_study_data['fixed_enhanced']['fail_Q_factor'] is None and error_fixed_enhanced >= FAIL_THRESHOLD_EUROS:
                enhanced_study_data['fixed_enhanced']['fail_Q_factor'] = Q_factor
                enhanced_study_data['fixed_enhanced']['fail_PSNR'] = PSNR
                enhanced_study_data['fixed_enhanced']['fail_TC'] = TC
                enhanced_study_data['fixed_enhanced']['fail_img_bgr'] = comp_res_fixed_enhanced['image']

            # Détection de la défaillance (CALIBRÉE enhanced)
            if enhanced_study_data['calibrated_enhanced']['fail_Q_factor'] is None and error_calibrated_enhanced >= FAIL_THRESHOLD_EUROS:
                enhanced_study_data['calibrated_enhanced']['fail_Q_factor'] = Q_factor
                enhanced_study_data['calibrated_enhanced']['fail_PSNR'] = PSNR
                enhanced_study_data['calibrated_enhanced']['fail_TC'] = TC
                enhanced_study_data['calibrated_enhanced']['fail_img_bgr'] = comp_res_calibrated_enhanced['image']

        # --- 1. Rapports Détaillés par Méthode ---
        report_fixed = generate_method_report_compression(fixed_standard, image_name, orig_value, "Standard - Contraintes Fixes", detect_circles_fixed)
        report_calibrated = generate_method_report_compression(calibrated_standard, image_name, orig_value, "Standard - Calibrage Dynamique", detect_circles_calibrated)
        report_fixed_enhanced = generate_method_report_compression(enhanced_study_data['fixed_enhanced'], image_name, orig_value, "Amélioré - Contraintes Fixes", detect_circles_fixed)
        report_calibrated_enhanced = generate_method_report_compression(enhanced_study_data['calibrated_enhanced'], image_name, orig_value, "Amélioré - Calibrage Dynamique", detect_circles_calibrated)

        # --- 2. Tableau de comparaison des seuils ---

        
        methods_data = {
            "Contraintes Fixes": {
                "standard": {
                    "PSNR": fixed_standard['fail_PSNR'],
                    "TC": fixed_standard['fail_TC'],
                    "Q": fixed_standard['fail_Q_factor']
                },
                "enhanced": {
                    "PSNR": enhanced_study_data['fixed_enhanced']['fail_PSNR'],
                    "TC": enhanced_study_data['fixed_enhanced']['fail_TC'],
                    "Q": enhanced_study_data['fixed_enhanced']['fail_Q_factor']
                }
            },
            "Calibrage Dynamique": {
                "standard": {
                    "PSNR": calibrated_standard['fail_PSNR'],
                    "TC": calibrated_standard['fail_TC'],
                    "Q": calibrated_standard['fail_Q_factor']
                },
                "enhanced": {
                    "PSNR": enhanced_study_data['calibrated_enhanced']['fail_PSNR'],
                    "TC": enhanced_study_data['calibrated_enhanced']['fail_TC'],
                    "Q": enhanced_study_data['calibrated_enhanced']['fail_Q_factor']
                }
            }
        }

        
        comparison_html = f"""
        <h4>Synthèse des Seuils de Défaillance (Compression DCT) - Standard vs Amélioré</h4>
        <table style='width:100%'>
            <tr>
                <th>Méthode</th>
                <th>Variante</th>
                <th>Q-Facteur de Défaillance</th>
                <th>TC de Défaillance</th>
                <th>PSNR de Défaillance</th>
                <th>Amélioration</th>
            </tr>
        """

        
        for method_name, method_data in methods_data.items():
            
            std_Q = method_data["standard"]["Q"]
            std_TC = method_data["standard"]["TC"]
            std_PSNR = method_data["standard"]["PSNR"]

            enh_Q = method_data["enhanced"]["Q"]
            enh_TC = method_data["enhanced"]["TC"]
            enh_PSNR = method_data["enhanced"]["PSNR"]

            
            q_improvement = "-"
            tc_improvement = "-"
            if std_Q is not None and enh_Q is not None:
                q_improvement = f"+{((enh_Q / std_Q) - 1) * 100:.1f}%"

            if std_TC is not None and enh_TC is not None:
                tc_improvement = f"+{((enh_TC / std_TC) - 1) * 100:.1f}%"

            
            std_Q_str = f"{std_Q:.2f}" if std_Q is not None else "N/A"
            std_TC_str = f"{std_TC:.2f}" if std_TC is not None else "N/A"
            std_PSNR_str = f"{std_PSNR:.2f} dB" if std_PSNR is not None else "N/A"

            enh_Q_str = f"{enh_Q:.2f}" if enh_Q is not None else "N/A"
            enh_TC_str = f"{enh_TC:.2f}" if enh_TC is not None else "N/A"
            enh_PSNR_str = f"{enh_PSNR:.2f} dB" if enh_PSNR is not None else "N/A"

            
            comparison_html += f"""
            <tr>
                <td rowspan="2">{method_name}</td>
                <td>Standard</td>
                <td>{std_Q_str}</td>
                <td>{std_TC_str}</td>
                <td>{std_PSNR_str}</td>
                <td>-</td>
            </tr>
            """

           
            comparison_html += f"""
            <tr>
                <td>Amélioré</td>
                <td style="color: {'green' if q_improvement != '-' and float(q_improvement[1:-1]) > 0 else 'inherit'}">{enh_Q_str}</td>
                <td style="color: {'green' if tc_improvement != '-' and float(tc_improvement[1:-1]) > 0 else 'inherit'}">{enh_TC_str}</td>
                <td>{enh_PSNR_str}</td>
                <td style="font-weight: bold; color: {'green' if q_improvement != '-' and float(q_improvement[1:-1]) > 0 else 'inherit'}">{q_improvement}</td>
            </tr>
            """

        comparison_html += """
        </table>
        <p style="font-style:italic;">Une valeur <strong>Q</strong> plus élevée, <strong>TC</strong> plus élevée ou <strong>PSNR</strong> plus faible indique une plus grande robustesse à la compression.</p>
        """

        # --- Assemblage du HTML pour l'image ---
        results_html.append(f"""
            <div style="border: 2px solid #ccc; padding: 15px; margin-bottom: 20px;">
                <h2>Problème 3 Amélioré: Étude de la Compression DCT - {image_name}</h2>
                {comparison_html}
                <h3>Méthodes Standards</h3>
                <div style="display: flex; justify-content: space-around;">
                    <div style="width: 48%;"> {report_fixed} </div>
                    <div style="width: 48%;"> {report_calibrated} </div>
                </div>
                <h3>Méthodes Améliorées (Anti-DCT Artifacts)</h3>
                <div style="display: flex; justify-content: space-around;">
                    <div style="width: 48%;"> {report_fixed_enhanced} </div>
                    <div style="width: 48%;"> {report_calibrated_enhanced} </div>
                </div>
            </div>
        """)

    
    fail_PSNR_calibrated_final = enhanced_study_data['calibrated_enhanced']['fail_PSNR']
    fail_TC_calibrated_final = enhanced_study_data['calibrated_enhanced']['fail_TC']

    return "Étude de la compression terminée avec méthode améliorée. Comparaison effectuée.", "".join(results_html), fail_PSNR_calibrated_final, fail_TC_calibrated_final

# --- FONCTIONS DE GESTION DES SECTIONS GRADIO ---

def process_upload(files, state):
    """ Section 1 : Gestion de l'upload des images. """
    images_data = []
    image_names = []
    if files is not None:
        for f in files:
            try:
                # Utilisation de np.fromfile et cv2.imdecode pour la compatibilité avec Colab/Gradio
                file_bytes = np.fromfile(f.name, dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is not None:
                    images_data.append(img)
                    image_names.append(os.path.basename(f.name))
                else:
                    print(f"Erreur: Impossible de décoder l'image {os.path.basename(f.name)}. Vérifiez le format.")
            except Exception as e:
                filename_for_error = getattr(f, 'name', 'Fichier Inconnu')
                print(f"Erreur de chargement pour {filename_for_error}: {e}")

    if not images_data:
        return "Erreur : Aucune image valide chargée. Vérifiez les formats (JPG, PNG).", state

    state['images'] = images_data
    state['names'] = image_names
    state['results_prob1_fixed'] = []
    state['results_prob1_calibrated'] = []
    # Réinitialiser les autres états
    state['standard_study_data'] = None

    return f"**{len(images_data)} image(s) chargée(s). Prêt pour le Problème 1.**", state

def process_prob1(state):
    """
    Section 2 : Exécution des deux solutions (Fixe et Calibrée).
    """
    images_data = state.get('images', [])
    image_names = state.get('names', [])

    if not images_data:
        return "Erreur : Aucune image chargée. Veuillez d'abord exécuter la Section 1.", "", state

    results_html = []
    results_prob1_fixed = []
    results_prob1_calibrated = []

    for img_bgr, name in zip(images_data, image_names):

        # --- Méthode 1: Fixe ---
        img_drawn_fixed, final_count_text_fixed, results_fixed = detect_circles_fixed(img_bgr)
        results_fixed['name'] = name
        results_prob1_fixed.append(results_fixed)
        img_b64_fixed = encode_img_b64(img_drawn_fixed)

        # --- Méthode 2: Calibrée ---
        img_drawn_calibrated, final_count_text_calibrated, results_calibrated = detect_circles_calibrated(img_bgr)
        results_calibrated['name'] = name
        results_prob1_calibrated.append(results_calibrated)
        img_b64_cal = encode_img_b64(img_drawn_calibrated)

        results_html.append(f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 20px;">
                <h3>Résultats pour : {name}</h3>

                <div style="display: flex; justify-content: space-around;">
                    <div style="flex: 1; margin-right: 10px; border-right: 1px dashed #ccc; padding-right: 10px;">
                        <h4>Méthode 1: Contraintes Fixes (Rigide)</h4>
                        <p><strong>{final_count_text_fixed}</strong></p>
                        <img src="data:image/png;base64,{img_b64_fixed}" alt="Pièces fixes" style="max-width: 100%; height: auto;">
                    </div>
                    <div style="flex: 1; margin-left: 10px; padding-left: 10px;">
                        <h4>Méthode 2: Calibrage Dynamique (Robuste, sans Saturation)</h4>
                        <p><strong>{final_count_text_calibrated}</strong></p>
                        <img src="data:image/png;base64,{img_b64_cal}" alt="Pièces calibrées" style="max-width: 100%; height: auto;">
                    </div>
                </div>
            </div>
        """)

    state['results_prob1_fixed'] = results_prob1_fixed
    state['results_prob1_calibrated'] = results_prob1_calibrated

    output_text = "".join(results_html)
    return "✅ **Problème 1 terminé.** Diagnostic de base établi pour les deux méthodes.", output_text, state

def process_prob2(state):
    """ Section 3 : Problème 2 - Bruit Sel et Poivre. """
    images = state.get('images', [])
    results_fixed = state.get('results_prob1_fixed', [])
    results_calibrated = state.get('results_prob1_calibrated', [])

    if not images or not results_fixed or not results_calibrated:
        return "Erreur : Résultats du Problème 1 manquants. Veuillez exécuter les Sections 1 et 2.", "", "", ""

    status, html_output, fail_d, fail_Q = study_noise(images, results_fixed, results_calibrated)
    return status, html_output, fail_d, fail_Q

def process_prob3(state):
    """ Section 4 : Problème 3 - Compression DCT. """
    images = state.get('images', [])
    results_fixed = state.get('results_prob1_fixed', [])
    results_calibrated = state.get('results_prob1_calibrated', [])

    if not images or not results_fixed or not results_calibrated:
        return "Erreur : Résultats du Problème 1 manquants. Veuillez exécuter les Sections 1 et 2.", "", "", ""

    status, html_output, fail_PSNR, fail_TC, standard_study_data = study_compression(images, results_fixed, results_calibrated)

    # Stocker les données d'étude standard pour réutilisation dans le problème 3 amélioré
    state['standard_study_data'] = standard_study_data

    return status, html_output, fail_PSNR, fail_TC, state

def process_prob3_enhanced(state):
    """
    Section 5 : Problème 3 Amélioré - Compression DCT avec anti-artifacts.
    Réutilise les données du problème 3 standard.
    """
    images = state.get('images', [])
    results_fixed = state.get('results_prob1_fixed', [])
    results_calibrated = state.get('results_prob1_calibrated', [])
    standard_study_data = state.get('standard_study_data', None)

    if not images or not results_fixed or not results_calibrated:
        return "Erreur : Résultats du Problème 1 manquants. Veuillez exécuter les Sections 1 et 2.", "", "", ""

    if standard_study_data is None:
        return "Erreur : Résultats du Problème 3 manquants. Veuillez d'abord exécuter la Section 4.", "", "", ""

    status, html_output, fail_PSNR, fail_TC = study_compression_enhanced(images, results_fixed, results_calibrated, standard_study_data)
    return status, html_output, fail_PSNR, fail_TC

# --- INTERFACE GRADIO ---

with gr.Blocks(title="EuroCount") as demo:

    # gr.State initialisé pour stocker les résultats des deux méthodes et données d'étude
    state = gr.State({
        'images': [],
        'names': [],
        'results_prob1_fixed': [],
        'results_prob1_calibrated': [],
        'standard_study_data': None
    })

    gr.Markdown("DM-Traitement Numerique d'Image")
    gr.HTML(f"""<h2>Le temps d'execution peut prendre quelques minutes pour une image----Alors pour 7 images Café obligatoire ☕🖤.</h2>
    <h3>Vous pouvez exécuter la section 4 avant que la section 3 ne soit terminée ! (Valable uniquement pour ces 2 sections)</h3>
       Auteur: <strong>{AUTHOR}</strong> </strong></p>
    """)

    # ----------------------------------------------------
    # SECTION 1: Importation des Images
    # ----------------------------------------------------
    with gr.Tab("1. Importation des Images"):
        gr.Markdown("## 1. Importation des Pièces de Monnaie (Originales)")
        file_input = gr.Files(label="Charger une ou plusieurs images (JPG, PNG)", file_count="multiple")
        upload_btn = gr.Button("Charger et Enregistrer les Images")
        upload_status = gr.Textbox(label="Statut du Chargement", lines=1)

        upload_btn.click(
            fn=process_upload,
            inputs=[file_input, state],
            outputs=[upload_status, state]
        )

    # ----------------------------------------------------
    # SECTION 2: Problème 1 (Comptage de Base)
    # ----------------------------------------------------
    with gr.Tab("2. Problème 1 : Comptage de Base "):
        gr.Markdown("## 2. Exécution du Comptage (Fixe vs. Calibrée)")
        prob1_btn = gr.Button("Exécuter le Comptage de Base pour les Deux Méthodes")
        prob1_status = gr.Textbox(label="Statut de l'Exécution", lines=1)
        prob1_output_html = gr.HTML(label="Résultats Détaillés du Problème 1 (Comparaison Visuelle)")

        prob1_btn.click(
            fn=process_prob1,
            inputs=[state],
            outputs=[prob1_status, prob1_output_html, state]
        )

    # ----------------------------------------------------
    # SECTION 3: Problème 2 (Bruit Sel et Poivre)
    # ----------------------------------------------------
    with gr.Tab("3. Problème 2 : Bruit Sel et Poivre "):
        gr.Markdown("## 3. Étude du Bruit Impulsionnel (Comparaison de Robustesse, Filtres et Images Clés)")
        prob2_btn = gr.Button("Lancer l'Analyse Détaillée du Bruit (d et Q)")
        prob2_status = gr.Textbox(label="Statut de l'Analyse", lines=1)

        with gr.Row():
            fail_d_out = gr.Textbox(label="d de Défaillance (Méthode Calibrée)", interactive=False)
            fail_Q_out = gr.Textbox(label="Q de Défaillance (Méthode Calibrée)", interactive=False)

        prob2_output_html = gr.HTML(label="Rapport Complet (Graphiques, Tableaux de Comparaison, Filtres et Images Corrigées)")

        prob2_btn.click(
            fn=process_prob2,
            inputs=[state],
            outputs=[prob2_status, prob2_output_html, fail_d_out, fail_Q_out]
        )

    # ----------------------------------------------------
    # SECTION 4: Problème 3 (Compression DCT)
    # ----------------------------------------------------
    with gr.Tab("4. Problème 3 : Compression DCT "):
        gr.Markdown("## 4. Étude de la Compression (Comparaison de Robustesse Détaillée, Tableaux et Graphes)")
        prob3_btn = gr.Button("Lancer l'Analyse Détaillée de la Compression (TC et PSNR)")
        prob3_status = gr.Textbox(label="Statut de l'Analyse", lines=1)

        with gr.Row():
            fail_psnr_out = gr.Textbox(label="PSNR de Défaillance (Méthode Calibrée)", interactive=False)
            fail_tc_out = gr.Textbox(label="TC de Défaillance (Méthode Calibrée)", interactive=False)

        prob3_output_html = gr.HTML(label="Rapport Complet (Graphiques, Tableaux de Balayage et Images à la Défaillance)")

        prob3_btn.click(
            fn=process_prob3,
            inputs=[state],
            outputs=[prob3_status, prob3_output_html, fail_psnr_out, fail_tc_out, state]
        )

    # ----------------------------------------------------
    # SECTION 5: Problème 3 Compression DCT avec anti-artifacts
    # ----------------------------------------------------
    with gr.Tab("5. Problème 3 Compression DCT avec Anti-Artifacts "):
        gr.Markdown("## 5. Étude de la Compression avec Traitement Anti-Artifacts DCT")
        gr.HTML("""
            <div style="background-color: #e6f7ff; border-left: 4px solid #1890ff; padding: 10px; margin-bottom: 20px;">
                <p><strong>Nouvelle fonctionnalité!</strong> Cette section ajoute un prétraitement spécial pour réduire les artefacts
                de compression DCT et améliorer la détection des pièces même sous forte compression.</p>
                <p>Le traitement adaptatif inclut:</p>
                <ul>
                    <li>Détection et estimation automatique du niveau de compression</li>
                    <li>Filtrage bilatéral adaptatif pour réduire les effets de bloc</li>
                    <li>Renforcement des contours préservant la détection des cercles</li>
                    <li>Mélange intelligent des zones lisses/texturées pour éviter les faux positifs</li>
                </ul>
                <p><strong>Important:</strong> Cette section utilise les mêmes images compressées que le Problème 3 standard, puis leur applique
                le traitement anti-artifacts, garantissant ainsi une comparaison directe et équitable des méthodes.</p>
            </div>
        """)

        prob3_enhanced_btn = gr.Button("Lancer l'Analyse Détaillée de la Compression Améliorée")
        prob3_enhanced_status = gr.Textbox(label="Statut de l'Analyse", lines=1)

        with gr.Row():
            fail_psnr_enhanced_out = gr.Textbox(label="PSNR de Défaillance (Méthode Améliorée)", interactive=False)
            fail_tc_enhanced_out = gr.Textbox(label="TC de Défaillance (Méthode Améliorée)", interactive=False)

        prob3_enhanced_output_html = gr.HTML(label="Rapport Complet (Comparaison Standard vs Amélioré)")

        prob3_enhanced_btn.click(
            fn=process_prob3_enhanced,
            inputs=[state],
            outputs=[prob3_enhanced_status, prob3_enhanced_output_html, fail_psnr_enhanced_out, fail_tc_enhanced_out]
        )

if __name__ == "__main__":
    demo.launch(debug=True)


