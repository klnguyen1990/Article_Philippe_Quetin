import os
import numpy as np
import nibabel as nib
import cv2
from medpy.metric.binary import dc
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import wilcoxon
import warnings
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt
basename = os.path.basename
splitext = os.path.splitext

warnings.filterwarnings("ignore")


def restore_original_shape(vol_processed, original_shape, size=320):
    """
    Restores a processed volume back to its original shape.

    Parameters:
        vol_processed: np.ndarray — the processed volume of shape (size, size, size)
        original_shape: tuple — the original volume shape before any processing
        size: int — the crop/resize size used during preprocessing (default: 320)

    Returns:
        np.ndarray — volume restored to original_shape
    """
    pad_width = ((300, 300), (300, 300), (300, 300))
    padded_shape = tuple(s + 300 + 300 for s in original_shape)  # shape after padding

    # Step 1: Resize back to the cropped size before resizing
    center = (np.array(padded_shape) / 2).astype(np.int32)
    cropped_size = tuple(min(2 * size, padded_shape[i]) for i in range(3))
    vol = resize(vol_processed, cropped_size, anti_aliasing=False, order=0)

    # Step 2: Place the cropped volume back into the padded volume
    vol_padded = np.zeros(padded_shape, dtype=np.uint8)
    vol_padded[
        center[0] - size:center[0] + size,
        center[1] - size:center[1] + size,
        center[2] - size:center[2] + size
    ] = vol

    # Step 3: Remove the padding
    vol_restored = vol_padded[
        pad_width[0][0]: padded_shape[0] - pad_width[0][1],
        pad_width[1][0]: padded_shape[1] - pad_width[1][1],
        pad_width[2][0]: padded_shape[2] - pad_width[2][1]
    ]

    return vol_restored.astype(np.uint8)

# ==========================================================
# CURVATURE (SDF-BASED)
# ==========================================================
def compute_sdf(binary):
    dist_out = distance_transform_edt(binary == 0)
    dist_in = distance_transform_edt(binary == 1)
    return dist_out - dist_in


def compute_mean_curvature(phi, eps=1e-8):
    gx, gy, gz = np.gradient(phi)
    grad_norm = np.sqrt(gx**2 + gy**2 + gz**2 + eps)

    nx, ny, nz = gx / grad_norm, gy / grad_norm, gz / grad_norm

    nxx = np.gradient(nx, axis=0)
    nyy = np.gradient(ny, axis=1)
    nzz = np.gradient(nz, axis=2)

    return nxx + nyy + nzz


def curvature_smoothness_voxel(binary, sigma=0.0):
    if np.sum(binary) < 10:
        return {"mean_abs_curvature": 0.0}, None, None

    sdf = compute_sdf(binary)

    H = compute_mean_curvature(sdf)

    surface_mask = np.abs(sdf) < 1.5
    H_surface = H[surface_mask]

    if len(H_surface) == 0:
        return {"mean_abs_curvature": 0.0}, H, None

    metrics = {
        "mean_curvature_mean": np.mean(H_surface),
        "mean_curvature_std": np.std(H_surface),
        "mean_abs_curvature": np.mean(np.abs(H_surface)),
    }

    return metrics, H, None


# ==========================================================
# OPTION C — Z-AXIS CONSISTENCY (NO NORMALIZATION)
# ==========================================================
def z_axis_consistency(sdf, band=1.5):
    """
    Measures inter-slice continuity using raw Z-gradient of SDF.
    """
    gz = np.gradient(sdf, axis=1)

    surface_mask = np.abs(sdf) < band
    gz_surface = gz[surface_mask]

    if len(gz_surface) == 0:
        return 0.0

    return np.mean(np.abs(gz_surface))


# ==========================================================
# HAUSDORFF / MAD (UNCHANGED)
# ==========================================================
def custom_hausdorff_mad_3d(p, g, spacing_pred, spacing_gt):
    from scipy.ndimage import binary_erosion, generate_binary_structure
    from scipy.spatial import KDTree

    struct = generate_binary_structure(3, 1)  # 6-connectivity 3D neighbourhood
    edges_p = p.astype(bool) & ~binary_erosion(p, struct)
    edges_g = g.astype(bool) & ~binary_erosion(g, struct)

    coord_p = np.argwhere(edges_p).astype(float)
    coord_g = np.argwhere(edges_g).astype(float)

    if coord_p.shape[0] == 0 or coord_g.shape[0] == 0:
        return 100.0, 100.0

    coord_p *= spacing_pred
    coord_g *= spacing_gt

    tree_p = KDTree(coord_p)
    tree_g = KDTree(coord_g)

    d_g2p, _ = tree_p.query(coord_g)
    d_p2g, _ = tree_g.query(coord_p)

    Hausdorff_max = max(d_g2p.max(), d_p2g.max())
    final_mad = (d_g2p.mean() + d_p2g.mean()) / 2

    return Hausdorff_max, final_mad


# ==========================================================
# MAIN METRICS FUNCTION (UPDATED)
# ==========================================================
def compute_all_metrics(pred_path, gt_path, original_shape, ps_gt, class_id=2):
    pred_nib, gt_nib = nib.load(pred_path), nib.load(gt_path)
    spacing = np.array(gt_nib.header.get_zooms()) * ps_gt

    p = (pred_nib.get_fdata() == 1).astype(int)
    g = (gt_nib.get_fdata() == class_id).astype(int)

    p_o = restore_original_shape(p,original_shape)
    """# 1. Create a default identity affine matrix (required by NIfTI format)
    affine = np.eye(4)

    # 2. Convert the NumPy array (p_o) into a NIfTI image object
    nifti_img = nib.Nifti1Image(p_o, affine)

    # 3. Save the image object
    nib.save(nifti_img, 'output_volume.nii.gz')"""

    sum_p, sum_g = float(np.sum(p)), float(np.sum(g))
    if sum_g == 0:
        return None

    # -------------------------
    # OVERLAP
    # -------------------------
    dice = dc(p, g)
    delta_v_r = ((sum_p - sum_g) / sum_p) * 100 

    # -------------------------
    # SURFACE GEOMETRY
    # -------------------------
    if sum_p > 0:
        try:
            hd_val, mad_val = custom_hausdorff_mad_3d(p, g, spacing, spacing)
        except:
            hd_val, mad_val = 100.0, 100.0

        sdf = compute_sdf(p)

        curv_metrics, _, _ = curvature_smoothness_voxel(p)

        # OPTION C: Z-axis consistency
        z_grad = z_axis_consistency(sdf)

    else:
        hd_val, mad_val = 100.0, 100.0
        curv_metrics = {"mean_abs_curvature": 0.0}
        z_grad = 0.0

    return {
        "Dice": dice,
        "DeltaVR": np.abs(delta_v_r),
        "Volume" : float(np.sum(p_o))*(ps_gt**3),
        "HD": hd_val,
        "MAD": mad_val,
        "Curv_Abs": curv_metrics["mean_abs_curvature"],
        "Z_Grad_Abs": z_grad
    }


# ==========================================================
# BATCH PROCESSING
# ==========================================================
#pred_dir = r"C:\Users\nguyen\Desktop\Neobrain\DATASET_Neobrain\unetr_pp_results\inferTs"
pred_dir = r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\vnet_VL&TL_trained\results"
gt_dir = r"C:\Users\nguyen\Desktop\DATASET_Neobrain\unetr_pp_raw\unetr_pp_raw_data\Task002_Neobrain\labelsTs"

#files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii")])
all_res = []

h = f"{'Patient':<20} | {'Dice':<6} | {'ΔVr %':<8} | {'Volume':<6} | {'HD':<6} | {'MAD':<6} | {'C_Abs':<7} | {'Z_Grad':<8}"
print("\n" + h)
print("-" * len(h))

"""ps_pred = {"Patient37_J10_49":0.141343,
      "Patient48_J12_43":0.150177,
      "Patient49_J12_22":0.150177,
      "Patient53_J17_41":0.150177,
      "Patient81_J24_62":0.159011
      }"""

ps_gt = {"Patient37_J10_49":0.150177,
      "Patient48_J12_43":0.150177,
      #"Patient49_J12_22":0.159011,
      "Patient53_J17_41":0.159011,
      "Patient81_J24_62":0.167844
      }
original_shape = {"Patient37_J10_49":[759, 849, 810],
      "Patient48_J12_43":[739, 830, 703],
      #"Patient49_J12_22":[],
      "Patient53_J17_41":[749, 942, 784],
      "Patient81_J24_62":[744, 919, 803]

}

for f in files:
    if splitext(splitext(basename(f))[0])[0] != "Patient49_J12_22":
        #res = compute_all_metrics(os.path.join(pred_dir, f), os.path.join(gt_dir, f), original_shape[splitext(splitext(basename(f))[0])[0]], ps_gt[splitext(splitext(basename(f))[0])[0]])
        res = compute_all_metrics(os.path.join(pred_dir, f), os.path.join(gt_dir, f.replace('.nii','.nii.gz')),original_shape[splitext(basename(f))[0]], ps_gt[splitext(basename(f))[0]])
    
        if res:
            all_res.append(res)
            print(f"{f[:20]:<20} | {res['Dice']:6.2f} | {res['DeltaVR']:8.2f} | "
                  f"{res['Volume']:6.2f} | "
                  f"{res['HD']:6.2f} | {res['MAD']:6.2f} | "
                  f"{res['Curv_Abs']:7.2f} | {res['Z_Grad_Abs']:8.4f}")


# ==========================================================
# SUMMARY + STATISTICS
# ==========================================================
if all_res:
    print("-" * len(h))

    keys = ["Dice", "HD", "MAD", "DeltaVR", "Volume", "Curv_Abs", "Z_Grad_Abs"]
    stats = {}

    for k in keys:
        values = [r[k] for r in all_res]
        stats[k] = (np.mean(values), np.std(values))

    print(f"{'MEAN ± STD':<20} | "
          f"{stats['Dice'][0]:.3f}±{stats['Dice'][1]:.3f} | "
          f"{stats['DeltaVR'][0]:8.2f}±{stats['DeltaVR'][1]:.2f} | "
          f"{stats['Volume'][0]:.3f}±{stats['Volume'][1]:.3f} | "
          f"{stats['HD'][0]:6.2f}±{stats['HD'][1]:.2f} | "
          f"{stats['MAD'][0]:6.2f}±{stats['MAD'][1]:.2f} | "
          f"{stats['Curv_Abs'][0]:.4f}±{stats['Curv_Abs'][1]:.4f} | "
          f"{stats['Z_Grad_Abs'][0]:.4f}±{stats['Z_Grad_Abs'][1]:.4f}")

    print("-" * len(h))