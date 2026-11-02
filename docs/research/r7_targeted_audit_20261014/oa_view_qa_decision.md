# R7 OA compartment view QA, 14 October 2026

The private CPU Kaggle [pixel-view pilot](oa_view_output/oa_view_pilot/receipt.json)
decoded 360 sampled slices from 72 series in 24 fixed, non-Gold studies. All
cases had sagittal, coronal, and axial views. The orange 140 mm field in each
[montage](oa_view_output/oa_view_pilot/case_01.jpg) is the approximate field
used by earlier broad readers. The cyan boxes in the
[ROI overlays](oa_roi_overlay/case_01_roi.jpg) are proposed fixed physical
views. No lesion truth or AUC was measured.

In these 24 studies, 10 had a laterality DICOM tag and 14 required patient-X
coordinate inference. All 10 tags agreed with that inference. A visual review
found that the patellofemoral axial boxes covered the patella/trochlear region
at one or more sampled levels in all 12 PF cases. Lateral coronal boxes covered
the lateral joint-line region in the 12 lateral cases. The initial lateral
sagittal two-position overlay missed visible femorotibial cartilage in case 3;
the lateralmost sampled image showed it. This is why the feature probe uses
three consecutive windows at the physical lateral end of the sagittal series,
not the overlay's original q50/q70 choice. These are engineering observations
from downsampled contact sheets, **not** expert cartilage annotations or a
claim that a lesion falls inside each box.

The ROI experiment is therefore technically worth a bounded test, with the
following fixed comparison: frozen ImageNet ConvNeXt-Tiny on identical cached
windows, whole 140 mm view versus the proposed 65–88 mm OA views. The same
small adjacent-window head and grouped training protocol will score both
arms. Every Gold58-linked report/patient group is excluded from fit and
validation. Clear weak-report labels are the primary proxy endpoint; this
cannot prove clinical or public-score improvement. The earlier SKM-TEA mask
transfer failure still rules out using those masks as the reader input.
