import numpy as np
import nibabel as ni
import pyvista as pv
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib.colors import ListedColormap
from scipy.ndimage import convolve, center_of_mass, binary_dilation
from nilearn.image import resample_to_img



fig, axes = plt.subplots(1, 3)
fig.set_size_inches(18, 5)
fs = 21
plt.subplots_adjust(wspace=0.5, hspace=0.5)
axes[0].set_title("Sagittal view", fontsize=fs)
axes[1].set_title("Coronal view", fontsize=fs)
axes[2].set_title("Axial view", fontsize=fs)


def  Dice(X, Y):
    X = (X > 0).astype(np.int8)
    Y = (Y > 0).astype(np.int8)

    X_nb = np.sum(X.reshape(-1))
    Y_nb = np.sum(Y.reshape(-1))

    return (2 * (X * Y)) / (X_nb + Y_nb)


def LoadVolumes(volumes_path, spacing):
    print("Loading volumes...")
    vols = []

    for i, vol_path in enumerate(volumes_path):
        
        # Load volume
        nifti_vol = ni.load(vol_path)
        if i==0 : 
            nifti_vol_ref = nifti_vol
        if (np.array(nifti_vol.get_fdata()) > 0).shape==(320,320,320):
            nifti_vol = resample_to_img(nifti_vol, nifti_vol_ref, interpolation='nearest')
        vol = (np.array(nifti_vol.get_fdata()) > 0).astype(np.uint8)

        # Adjust to real size
        final_size = np.array(vol.shape) * (spacing[i] / spacing[0])
        vol = (resize(vol, final_size, anti_aliasing=False) > 0).astype(np.uint8)
        vol = np.pad(vol, ((150, 150), (150, 150), (150, 150)), mode='constant', constant_values=0)

        # center on center of mass of thalami with size 400^3
        com = np.array(center_of_mass(vol)).astype(np.int32)
        vol = vol[com[0]-200:com[0]+200, com[1]-200:com[1]+200, com[2]-200:com[2]+200]
        vols.append(vol)

    return vols


def GetSlices(labels):
    slices = [[], [], []]

    for i in range(3):
        if i == 1: # coronal view
            for k in range(len(labels)):
                labels[k] = np.swapaxes(labels[k], 0, 1)

        if i == 2: # axial view
            for k in range(len(labels)):
                labels[k] = np.swapaxes(labels[k], 0, 2)

        # La boucle k doit être INDENTÉE ici pour chaque vue i
        for k in range(len(labels)):
            if i == 0:
                slice_data = labels[k][88] # Worst : 283; best : 88
            else:
                slice_data = labels[k][199]

            kernel = np.ones((3, 3), dtype=int)
            kernel[1, 1] = 0
            label_neighbour_count = convolve(slice_data, kernel, mode='constant', cval=0)
            label = ((np.logical_not(slice_data) == 1) & (label_neighbour_count > 0)).astype(np.int8)
            label = np.swapaxes(label, 0, 1)
            label = binary_dilation(label, iterations=1)
            label = (label > 0).astype(np.int8)

            slices[i].append(label)
    
    return slices


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def Plot(labels, spacing_val):
    slices = GetSlices(labels)
    # Couleurs : Fond, Ref, Unet, Vnet, Unetr++
    cmap = ListedColormap(['black', 'white', 'lime', 'red', 'blue'])
    
    # --- SOLUTION : On ferme les éventuelles figures fantômes et on crée LA figure ---
    plt.close('all') # Ferme les fenêtres vides comme ta "Figure 1"
    fs = 50
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Crée la "Figure 1" avec tes 3 axes
    # --------------------------------------------------------------------------------

    for i in [1, 2, 0]:
        # Initialisation du canvas pour la vue i
        total = np.zeros_like(slices[i][0])

        for k in range(len(labels)):
            # On superpose les étiquettes (k+1 pour mapper vers la colormap)
            total = np.maximum(total, slices[i][k] * (k+1))

        # IMPORTANT : On définit l'index d'affichage pour respecter l'ordre Coronal, Axial, Sagittal
        # On fait correspondre l'index de donnée (i) à l'index de l'axe (plot_idx)
        if i == 1: plot_idx = 0; title = "Coronal view"
        elif i == 2: plot_idx = 1; title = "Axial view"
        else: plot_idx = 2; title = "Sagittal view"

        axes[plot_idx].imshow(total, cmap=cmap)
        axes[plot_idx].set_title(title, fontsize=fs + 2)

        if i == 1: # coronal
            axes[plot_idx].set_xlabel("Z Scale (mm)", fontsize=fs)
            axes[plot_idx].set_ylabel("Y Scale (mm)", fontsize=fs)
        elif i == 2: # axial
            axes[plot_idx].set_xlabel("Z Scale (mm)", fontsize=fs)
            axes[plot_idx].set_ylabel("X Scale (mm)", fontsize=fs)
        elif i == 0: # sag
            axes[plot_idx].set_xlabel("X Scale (mm)", fontsize=fs)
            axes[plot_idx].set_ylabel("Y Scale (mm)", fontsize=fs)

        # Mise à l'échelle des axes
        # Axe X
        nb_x = np.arange(0, slices[i][0].shape[1] * spacing_val)
        x1 = np.array([x for x in nb_x if x % 20 == 0]).astype(np.int32)
        x2 = x1 / spacing_val
        axes[plot_idx].set_xticks(x2)
        axes[plot_idx].set_xticklabels(x1, fontsize=fs)

        # Axe Y
        nb_y = np.arange(0, slices[i][0].shape[0] * spacing_val)
        y1 = np.array([y for y in nb_y if y % 20 == 0]).astype(np.int32)
        y2 = y1 / spacing_val
        axes[plot_idx].set_yticks(y2)
        axes[plot_idx].set_yticklabels(y1, fontsize=fs)
        
        axes[plot_idx].invert_yaxis()

    plt.tight_layout()
    plt.show()



def Plot3DSegmentations(labels):
    print("Showing volumes...")

    p = pv.Plotter()
    p.set_background('black')
    colors = ['white', 'lime', 'red', 'blue']
    

    for i in range(len(labels)):
        grid = pv.ImageData()
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        grid.dimensions = np.array(labels[i].shape) + 1
        grid.cell_data["values"] = labels[i].flatten(order="F")
        thresholded = grid.threshold(0.5)
        if i in [1,2,3]:
            p.add_mesh(thresholded, opacity=0.9, color=colors[i], smooth_shading=True, lighting=True, show_edges=False)
        else:
            p.add_mesh(thresholded, opacity=1, color=colors[i], smooth_shading=True, lighting=True, show_edges=False)


    p.camera_position = 'yx'
    p.camera.azimuth = 160
    p.camera.elevation = 20

    # Exemple avec PyVista (p étant ton plotter)
    # Remplace la ligne 194 par ceci :
    p.add_axes(
        line_width=5, 
        color='white',   # <--- TRÈS IMPORTANT : Blanc pour fond noir
        cone_radius=0.6, 
        shaft_length=0.7, 
        tip_length=0.3, 
        ambient=0.5, 
        label_size=(0.4, 0.16), # Taille ajustée pour être visible
        xlabel="Y", 
        ylabel="Z", 
        zlabel="X",
        labels_off=False   # <--- Force l'affichage
    )
    p.show()




if __name__ == "__main__":

    # Taille de voxel en mm correspondant aux volumes
    #spacing = [0.16, 0.19472, 0.16587, 0.16] # worst : 0.19472 best : 0.16
    spacing = [0.16, 0.16, 0.16587, 0.16]

    # Les trois volumes à comparer en .nii.gz
    labels = [
            r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLMoinsBonne\Patient81_J24_62_ref.nii.gz",
            r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLMoinsBonne\Patient81_J24_62_Unet.nii.gz",
            r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLMoinsBonne\Patient81_J24_62_Vnet.nii.gz",
            r"C:\Users\nguyen\Desktop\Neobrain\DATASET_Neobrain\unetr_pp_results\inferTs\Patient81_J24_62.nii.gz"
    ]

    labels = [
            r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLaMeilleure\Patient53_J17_41_ref.nii.gz",
        r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLaMeilleure\Patient53_J17_41_Unet_TL.nii.gz",
        r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Unet_Vnet_comparaison\SegAutoLaMeilleure\Patient53_J17_41_Vnet_TL.nii.gz",
        r"C:\Users\nguyen\Desktop\Neobrain\DATASET_Neobrain\unetr_pp_results\inferTs\Patient53_J17_41.nii.gz"
    ]


    labels = LoadVolumes(labels, spacing)
    Plot(labels, spacing[0])
    Plot3DSegmentations(labels)