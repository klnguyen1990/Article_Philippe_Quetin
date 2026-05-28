import numpy as np
import nibabel as ni
import pyvista as pv
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.colors import ListedColormap
from scipy.ndimage import convolve, center_of_mass, binary_dilation

# Initialisation de la figure globale
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(18, 5)
fs = 21
plt.subplots_adjust(wspace=0.5, hspace=0.5)
axes[0].set_title("Sagittal view", fontsize=fs)
axes[1].set_title("Coronal view", fontsize=fs)
axes[2].set_title("Axial view", fontsize=fs)

def Dice(X, Y):
    X = (X > 0).astype(np.int8)
    Y = (Y > 0).astype(np.int8)
    X_nb = np.sum(X)
    Y_nb = np.sum(Y)
    return (2 * np.sum(X * Y)) / (X_nb + Y_nb)

from nilearn.image import resample_to_img

def LoadVolumes(volumes_path, spacing):
    print("Loading volumes with physical resampling...")
    vols = []
    
    # 1. Charger la référence (Blanc) qui servira de grille cible
    ref_img = ni.load(volumes_path[0])
    
    for i, vol_path in enumerate(volumes_path):
        current_img = ni.load(vol_path)
        
        # 2. Resample physique : projette le volume i sur la grille de la référence
        # Cela règle l'échelle, l'orientation et la position automatiquement
        resampled_img = resample_to_img(current_img, ref_img, interpolation='nearest')
        
        vol = resampled_img.get_fdata()
        vol = (vol > 0).astype(np.uint8)

        # 3. Padding constant pour le centrage
        vol = np.pad(vol, ((150, 150), (150, 150), (150, 150)), mode='constant')

        # 4. Centrage sur le centre de masse
        com = np.array(center_of_mass(vol)).astype(np.int32)
        # On découpe un cube de 400x400x400 autour du centre de masse
        vol = vol[com[0]-200:com[0]+200, com[1]-200:com[1]+200, com[2]-200:com[2]+200]
        
        vols.append(vol)

    return vols

def GetSlices(labels):
    # On prépare 3 listes pour les 3 vues (Sagittal, Coronal, Axial)
    slices = [[], [], []]

    # Copie locale pour éviter de modifier les volumes originaux par swapaxes
    labels_copy = [np.copy(l) for l in labels]

    for i in range(3): # 0: Sagittal, 1: Coronal, 2: Axial
        if i == 1: # coronal view
            for k in range(len(labels_copy)):
                labels_copy[k] = np.swapaxes(labels_copy[k], 0, 1)
        if i == 2: # axial view
            for k in range(len(labels_copy)):
                labels_copy[k] = np.swapaxes(labels_copy[k], 0, 2)

        for k in range(len(labels_copy)):
            # Sélection de la coupe
            if i == 0:
                slice_data = labels_copy[k][250]
            else:
                slice_data = labels_copy[k][199]

            # Extraction des contours
            kernel = np.ones((3, 3), dtype=int)
            kernel[1, 1] = 0
            label_neighbour_count = convolve(slice_data, kernel, mode='constant', cval=0)
            
            # Détection des bords (pixel vide avec voisin plein)
            label = ((np.logical_not(slice_data) == 1) & (label_neighbour_count > 0)).astype(np.int8)
            label = np.swapaxes(label, 0, 1)
            label = binary_dilation(label, iterations=1)
            label = (label > 0).astype(np.int8)

            slices[i].append(label)
    
    return slices

def Plot(labels, spacing_val):
    slices = GetSlices(labels)
    # Couleurs : Fond, Ref, Unet, Vnet, Unetr++
    cmap = ListedColormap(['black', 'white', 'lime', 'red', 'blue'])

    for i in [1, 2, 0]:
        # Initialisation du canvas pour la vue i
        total = np.zeros_like(slices[i][0])

        for k in range(len(labels)):
            # On superpose les étiquettes (k+1 pour mapper vers la colormap)
            total = np.maximum(total, slices[i][k] * (k+1))

        axes[i].imshow(total, cmap=cmap)

        if i == 1: # coronal
            axes[i].set_xlabel("X Scale (mm)", fontsize=fs)
            axes[i].set_ylabel("Y Scale (mm)", fontsize=fs)
        elif i == 2: # axial
            axes[i].set_xlabel("Z Scale (mm)", fontsize=fs)
            axes[i].set_ylabel("Y Scale (mm)", fontsize=fs)
        elif i == 0: # sag
            axes[i].set_xlabel("Z Scale (mm)", fontsize=fs)
            axes[i].set_ylabel("X Scale (mm)", fontsize=fs)

        # Mise à l'échelle des axes
        # Axe X
        nb_x = np.arange(0, slices[i][0].shape[1] * spacing_val)
        x1 = np.array([x for x in nb_x if x % 20 == 0]).astype(np.int32)
        x2 = x1 / spacing_val
        axes[i].set_xticks(x2)
        axes[i].set_xticklabels(x1, fontsize=fs)

        # Axe Y
        nb_y = np.arange(0, slices[i][0].shape[0] * spacing_val)
        y1 = np.array([y for y in nb_y if y % 20 == 0]).astype(np.int32)
        y2 = y1 / spacing_val
        axes[i].set_yticks(y2)
        axes[i].set_yticklabels(y1, fontsize=fs)
        
        axes[i].invert_yaxis()

    plt.show()

def Plot3DSegmentations(labels):
    print("Showing volumes...")
    p = pv.Plotter()
    p.set_background('black')
    colors = ['white', 'green', 'red', 'blue']

    for i in range(len(labels)):
        grid = pv.ImageData()
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid.dimensions = np.array(labels[i].shape) + 1
        grid.cell_data["values"] = labels[i].flatten(order="F")
        thresholded = grid.threshold(0.5)
        p.add_mesh(thresholded, opacity=1, color=colors[i], smooth_shading=True, lighting=True, show_edges=False)

    p.camera_position = 'yx'
    p.camera.azimuth = 160
    p.camera.elevation = 20

    p.add_axes(line_width=5, color=(1, 1, 1), cone_radius=0.6, shaft_length=0.7, tip_length=0.3, ambient=0.5, 
                label_size=(0.4, 0.16), xlabel="Z", ylabel='X', zlabel='Y',)
    p.show()

if __name__ == "__main__":
    # Taille de voxel en mm
    spacing = [0.16, 0.19472, 0.16587, 0.16]

    # Chemins des fichiers
    
    #file_paths = [
    #    r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLMoinsBonne\Patient81_J24_62_ref.nii.gz",
    #    r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLMoinsBonne\Patient81_J24_62_Unet.nii.gz",
    #    r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLMoinsBonne\Patient81_J24_62_Vnet.nii.gz",
    #    r"C:\Users\nguyen\Desktop\Neobrain\DATASET_Neobrain\unetr_pp_results\inferTs\Patient81_J24_62.nii.gz"
    #]
    

    file_paths = [
        r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLaMeilleure\Patient53_J17_41_ref.nii.gz",
        r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLaMeilleure\Patient53_J17_41_Unet_TL.nii.gz",
        r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLaMeilleure\Patient53_J17_41_Vnet_TL.nii.gz",
        r"C:\Users\nguyen\Desktop\Neobrain\DATASET_Neobrain\unetr_pp_results\inferTs\Patient53_J17_41.nii.gz"
    ]

    # Exécution
    loaded_labels = LoadVolumes(file_paths, spacing)
    Plot(loaded_labels, spacing[0])
    Plot3DSegmentations(loaded_labels)