import os
import numpy as np
import nibabel as nib
import cv2
from medpy.metric.binary import dc
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import wilcoxon
import warnings
import matplotlib.pyplot as plt
basename = os.path.basename
splitext = os.path.splitext

warnings.filterwarnings("ignore")

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
def custom_hausdorff_mad_3d(p, g, spacing):
    MAD_ref2seg_total = 0
    MAD_seg2ref_total = 0
    Hausdorff_max = 0

    edges_p = np.zeros(p.shape, dtype=np.uint8)
    edges_g = np.zeros(g.shape, dtype=np.uint8)

    for i in range(g.shape[0]):
        edges_p[i,:,:] = cv2.Canny(np.uint8(p[i,:,:] * 255), 0, 0)
        edges_g[i,:,:] = cv2.Canny(np.uint8(g[i,:,:] * 255), 0, 0)

    coord_p = np.argwhere(edges_p).astype(float)
    coord_g = np.argwhere(edges_g).astype(float)

    if coord_p.shape[0] == 0 or coord_g.shape[0] == 0:
        return 100.0, 100.0

    coord_p *= spacing
    coord_g *= spacing

    for i in range(coord_g.shape[0]):
        d = np.sqrt(np.sum(np.square(coord_p - coord_g[i,:]), axis=1))
        min_d = np.min(d)
        MAD_ref2seg_total += min_d
        Hausdorff_max = max(Hausdorff_max, min_d)

    for i in range(coord_p.shape[0]):
        d = np.sqrt(np.sum(np.square(coord_g - coord_p[i,:]), axis=1))
        min_d = np.min(d)
        MAD_seg2ref_total += min_d
        Hausdorff_max = max(Hausdorff_max, min_d)

    final_mad = (MAD_ref2seg_total / coord_g.shape[0] +
                 MAD_seg2ref_total / coord_p.shape[0]) / 2
    return Hausdorff_max, final_mad


# ==========================================================
# MAIN METRICS FUNCTION (UPDATED)
# ==========================================================
def compute_all_metrics(pred_path, gt_path, ps_pred, ps_gt, class_id=2):
    pred_nib, gt_nib = nib.load(pred_path), nib.load(gt_path)
    #spacing = np.array(gt_nib.header.get_zooms()) * 0.15

    p = (pred_nib.get_fdata() == 1).astype(int)
    g = (gt_nib.get_fdata() == class_id).astype(int)

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
            hd_val, mad_val = custom_hausdorff_mad_3d(p, g, ps_gt)
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
        "Volume" : sum_p*(ps_pred**3),
        "HD": hd_val,
        "MAD": mad_val,
        "Curv_Abs": curv_metrics["mean_abs_curvature"],
        "Z_Grad_Abs": z_grad
    }


# ==========================================================
# BATCH PROCESSING
# ==========================================================
pred_dir = r"C:\Users\nguyen\Desktop\Neobrain\DATASET_Neobrain\unetr_pp_results\inferTs"
#pred_dir = r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\vnet_VL&TL_trained\results"
gt_dir = r"C:\Users\nguyen\Desktop\DATASET_Neobrain\unetr_pp_raw\unetr_pp_raw_data\Task002_Neobrain\labelsTs"

files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
#files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii")])
all_res = []

h = f"{'Patient':<20} | {'Dice':<6} | {'ΔVr %':<8} | {'Volume':<20} | {'HD':<6} | {'MAD':<6} | {'C_Abs':<7} | {'Z_Grad':<8}"
print("\n" + h)
print("-" * len(h))

ps_pred = {"Patient37_J10_49":0.0141343 ,
      "Patient48_J12_43":0.0150177 ,
      "Patient49_J12_22":0.0150177 ,
      "Patient53_J17_41":0.0150177 ,
      "Patient81_J24_62":0.0159011
      }

ps_gt = {"Patient37_J10_49":0.0150177 ,
      "Patient48_J12_43":0.0150177 ,
      "Patient49_J12_22":0.0159011 ,
      "Patient53_J17_41":0.0159011,
      "Patient81_J24_62":0.0167844
      }

for f in files:
    res = compute_all_metrics(os.path.join(pred_dir, f), os.path.join(gt_dir, f),ps_pred[splitext(splitext(basename(f))[0])[0]], ps_gt[splitext(splitext(basename(f))[0])[0]])
    #res = compute_all_metrics(os.path.join(pred_dir, f), os.path.join(gt_dir, f.replace('.nii','.nii.gz')),ps_pred[splitext(basename(f))[0]], ps_gt[splitext(basename(f))[0]])
    
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