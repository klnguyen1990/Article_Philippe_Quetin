import os
import numpy as np
import warnings
import nibabel as nib
import nrrd

basename = os.path.basename
splitext = os.path.splitext

warnings.filterwarnings("ignore")

# ==========================================================
# MAIN METRICS FUNCTION (UPDATED)
# ==========================================================
def compute_volumes(gt_path, ps_gt, tag, class_id=255):
    gt_nib, _ = nrrd.read(gt_path)
    g = (gt_nib == class_id).astype(int)
    sum_g = float(np.sum(g))
    if sum_g == 0:
        return None
    return {
        "RC": tag,
        "Volume" : sum_g*(ps_gt**3)
    }


# ==========================================================
# BATCH PROCESSING
# ==========================================================
gt_dir = r"C:\Users\nguyen\Desktop\postdoc_CREATIS_Lyon\Review article Philippe Quentin\Volumes+Contours_TableauComparatif_ICC"

# files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
dirs = sorted([d for d in os.listdir(gt_dir)])
all_res = []

h = f"{'Patient':<20} | {'RC':<6} | {'Volume':<6}"
print("\n" + h)
print("-" * len(h))


ps_gt = {"Patient6(37)_J10_49": 0.150177,
         "Patient8(48)_J12_43": 0.150177,
         "Patient10(50)_J12_43":0.150177,
         "Patient12(53)_J17_41": 0.159011,
         "Patient14(81)_J24_62": 0.167844
         }

for d in dirs:
    for sd in sorted(os.listdir(os.path.join(gt_dir, d))):
        if "R" in sd:
            for f in sorted(os.listdir(os.path.join(gt_dir, d, sd))):
                res = compute_volumes(os.path.join(gt_dir,d, sd, f), ps_gt[d], sd)

                if res:
                    all_res.append(res)
                    print(f"{d[:20]:<20} | {res['RC']:>6} | {res['Volume']:8.2f}")

    # ==========================================================
    # SUMMARY + STATISTICS
    # ==========================================================
    if all_res:
        print("-" * len(h))

        values = [r["Volume"] for r in all_res]
        vol_mean, vol_std = np.mean(values), np.std(values)

        print(f"{'MEAN ± STD':<20} | {'':>6} | {vol_mean:8.2f}±{vol_std:.2f}")

        print("-" * len(h))