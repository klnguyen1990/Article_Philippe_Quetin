import pyvista as pv
import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage

def clean_binary_volume(volume_3d):
    """
    Supprime les petits îlots de pixels isolés et remplit les cavités internes.
    """
    print("Nettoyage du volume (bruit et trous)...")
    
    # 1. Éliminer le bruit (ne garder que le plus gros objet)
    # On groupe les pixels connectés
    label_im, nb_labels = ndimage.label(volume_3d)
    if nb_labels > 1:
        # On calcule la taille de chaque groupe
        sizes = ndimage.sum(volume_3d, label_im, range(nb_labels + 1))
        # On trouve l'indice du plus gros groupe (en ignorant le fond à l'indice 0)
        largest_label = np.argmax(sizes[1:]) + 1
        # On crée un masque qui ne garde que ce groupe
        volume_3d = (label_im == largest_label).astype(np.uint8)
        print(f" -> {nb_labels - 1} petits objets isolés supprimés.")

    # 2. Remplir les trous internes (cavités dans l'os/organe)
    volume_3d = ndimage.binary_fill_holes(volume_3d).astype(np.uint8)
    print(" -> Cavités internes remplies.")
    
    return volume_3d

def convert_vtp_to_nifti_cleaned(input_vtp, output_nii, voxel_size=1.0):
    """
    Convertit un .vtp en NIfTI avec post-processing de nettoyage.
    """
    if not os.path.exists(input_vtp):
        print(f"Erreur : Le fichier {input_vtp} est introuvable.")
        return

    # --- ÉTAPE 1 : Chargement et Voxelisation ---
    mesh = pv.read(input_vtp)
    bounds = mesh.bounds
    origin = [bounds[0], bounds[2], bounds[4]]
    
    dims = [
        int(np.ceil((bounds[1] - bounds[0]) / voxel_size)) + 2,
        int(np.ceil((bounds[3] - bounds[2]) / voxel_size)) + 2,
        int(np.ceil((bounds[5] - bounds[4]) / voxel_size)) + 2
    ]

    grid = pv.ImageData()
    grid.dimensions = dims
    grid.spacing = [voxel_size] * 3
    grid.origin = origin

    # Sélection des points à l'intérieur
    selection = grid.select_enclosed_points(mesh, check_surface=True)
    mask_1d = selection['SelectedPoints']
    volume_np = mask_1d.reshape(dims, order='F') # Ordre X, Y, Z

    # --- ÉTAPE 2 : Nettoyage (Appel de la fonction de nettoyage) ---
    # On travaille sur le volume avant de changer l'ordre des axes pour SITK
    volume_np = clean_binary_volume(volume_np)

    # --- ÉTAPE 3 : Conversion et Sauvegarde ---
    # Transposer pour passer de (X, Y, Z) à (Z, Y, X) pour SimpleITK
    volume_sitk_format = np.transpose(volume_np, (2, 1, 0))
    
    sitk_img = sitk.GetImageFromArray(volume_sitk_format)
    sitk_img.SetSpacing([voxel_size] * 3)
    sitk_img.SetOrigin(origin)

    sitk.WriteImage(sitk_img, output_nii)
    print(f"Succès ! Volume nettoyé enregistré sous : {output_nii}")
    
    return volume_np

if __name__ == "__main__":
    # Remplacez par vos chemins de fichiers
    INPUT = r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Données_articles\volumes\Patient53_J17_41_ref.vtp"
    OUTPUT = "Patient53_cleaned.nii.gz"
    
    convert_vtp_to_nifti_cleaned(INPUT, OUTPUT, voxel_size=1.0)