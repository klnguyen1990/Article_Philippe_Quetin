import nibabel as ni
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
from skimage.transform import resize
from scipy.ndimage import center_of_mass

# 1. Chargement et Prétraitement
def LoadVolumes(volumes_path, spacing):
    print("Loading and preprocessing volumes...")
    vols = []
    nifti_vol_ref = None

    for i, vol_path in enumerate(volumes_path):
        # Load volume
        nifti_vol = ni.load(vol_path)
        
        if i == 0: 
            nifti_vol_ref = nifti_vol
        
        # Resampling si nécessaire
        if nifti_vol.get_fdata().shape == (320, 320, 320):
            nifti_vol = resample_to_img(nifti_vol, nifti_vol_ref, interpolation='nearest')
        
        if i == 3:
            # 1. Charger les données brutes (sans le > 0)
            vol = np.array(nifti_vol.get_fdata())

            # 2. Créer un masque où la valeur est exactement 2
            # Puis convertir en uint8 (0 et 1)
            vol = (vol == 2).astype(np.uint8)
            """plt.ion()
            plt.figure()
            plt.imshow(vol[:,:,vol.shape[2]//2], cmap='gray')
            plt.show()
            plt.pause(50)"""
        else:
            vol = (np.array(nifti_vol.get_fdata()) > 0).astype(np.uint8)

        # Adjust to real size (scaling)
        final_size = (np.array(vol.shape) * (spacing[i] / spacing[0])).astype(np.int32)
        vol = (resize(vol, final_size, anti_aliasing=False) > 0).astype(np.uint8)
        
        # Padding pour éviter les erreurs de bord lors du crop
        vol = np.pad(vol, ((150, 150), (150, 150), (150, 150)), mode='constant', constant_values=0)

        # Center on center of mass with size 400^3
        com = np.array(center_of_mass(vol)).astype(np.int32)
        vol = vol[com[0]-200:com[0]+200, com[1]-200:com[1]+200, com[2]-200:com[2]+200]
        vols.append(vol)

    return vols

def PlotContours(vols, names, slice_idx=None, fs=20, zoom_margin=20):
    # figsize hauteur augmentée pour accommoder les titres d'axes et Z_grad
    fig, axes = plt.subplots(1, len(names), figsize=(20, 8), facecolor='white') 
    colors = ['lime', 'red', 'blue', 'white'] # deepskyblue

    # --- ÉTAPE 1 : CALCUL DU ZOOM GLOBAL ---
    all_slices = []
    for vol in vols:
        curr = slice_idx if slice_idx is not None else vol.shape[2] // 2
        all_slices.append(np.rot90(vol[:, :, curr]))

    g_ymin, g_ymax = 1e6, 0
    g_xmin, g_xmax = 1e6, 0
    for s in all_slices:
        if np.any(s > 0):
            rows = np.any(s, axis=1)
            cols = np.any(s, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            g_ymin, g_ymax = min(g_ymin, ymin), max(g_ymax, ymax)
            g_xmin, g_xmax = min(g_xmin, xmin), max(g_xmax, xmax)

    # --- ÉTAPE 2 : AFFICHAGE ---
    for i, (ax, slice_data, name, color) in enumerate(zip(axes, all_slices, names, colors)):
        ax.imshow(np.zeros_like(slice_data), cmap='gray', vmin=0, vmax=1, origin='lower')

        if np.any(slice_data > 0):
            ax.contour(slice_data, levels=[0.5], colors=[color], linewidths=3)
            ax.set_ylim(g_ymin - zoom_margin, g_ymax + zoom_margin)
            ax.set_xlim(g_xmin - zoom_margin, g_xmax + zoom_margin)
        
        # Titre du modèle (en haut)
        ax.set_title(name, color='black', fontsize=fs+4, fontweight='bold', pad=15)
        
        # --- CONFIGURATION DES AXES (X et Z) ---
        ax.set_xlabel("Z", fontsize=fs, fontweight='bold')
        ax.set_ylabel("X", fontsize=fs, fontweight='bold')
        
        # On cache les chiffres (ticks) mais on garde les labels (X/Z)
        ax.set_xticks([])
        ax.set_yticks([])
        # On cache aussi la boîte de l'axe pour un look "minimaliste"
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        ax.set_aspect('equal')

        # --- ÉTAPE 3 : LOGIQUE DE DÉTECTION Z_GRAD ---
        n_clean = name.lower().replace("-", "")
        z_val = ""
        if "unet" in n_clean:
            # On met tout le bloc mathématique en gras
            z_val = "ASV = 0.65\n" + r"$\mathbf{\Delta V_r = 8.58}$ %"
        elif "vnet" in n_clean:
            z_val = "ASV = 0.57\n" + r"$\mathbf{\Delta V_r = 13.13}$ %"
        elif "segnet" in n_clean:
            z_val = "ASV = 0.57\n" + r"$\mathbf{\Delta V_r = 6.41}$ %"

        # L'annotation
        ax.annotate(z_val, xy=(0.5, -0.18), xycoords='axes fraction',
                    ha='center', va='top', fontsize=fs+2, 
                    # fontweight='bold' affectera "ASV = 0.65" mais pas le LaTeX
                    fontweight='bold', color='black')

    # Ajustement des marges : bottom=0.25 pour laisser de la place à "Z" ET "Z_grad"
    plt.subplots_adjust(wspace=0.2, left=0.08, right=0.95, top=0.85, bottom=0.25)
    plt.show()

# --- CONFIGURATION ET EXÉCUTION ---
volume_paths = [
    r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\unet_VL&TL_trained\results\Patient37_J10_49.nii",
    r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\vnet_VL&TL_trained\results\Patient37_J10_49.nii",
    r"C:\Users\nguyen\Desktop\Neobrain\DATASET_Neobrain\unetr_pp_results\inferTs\Patient37_J10_49.nii.gz",
    r"C:\Users\nguyen\Desktop\Neobrain\DATASET_Neobrain\unetr_pp_raw\unetr_pp_raw_data\Task002_Neobrain\labelsTs\Patient37_J10_49.nii.gz"
]

# Spacing exemple (à adapter selon tes données réelles)
spacings = [1, 1, 1, 1] 
model_names = ["U-Net", "V-Net", "ESegNet", "Ground truth"]

# 1. Charger
vols_processed = LoadVolumes(volume_paths, spacings)

# 2. Afficher
PlotContours(vols_processed, model_names, slice_idx=189)