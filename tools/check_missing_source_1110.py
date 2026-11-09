"""Validate the documented missing S2/OE source and restore its metadata only."""
from pathlib import Path
import argparse,json
import numpy as np

RID='2029_3764_13'
MEAN=np.array([1161.5256,1399.3857,1455.7269,2761.0648,1815.2145,2465.5625,2722.3400,2867.8205,2336.8220,1742.1495,1069.3466,3128.7744])
def verify(s2,oe):
 if s2.shape!=(24,256,256,12) or oe.shape!=(24,86,86,32):raise ValueError('Unexpected S2/OE cache shape')
 max_error=0.
 for t in range(24):
  a=np.asarray(s2[t],dtype=np.float32)
  if not np.isfinite(a).all() or not np.isfinite(oe[t]).all():raise ValueError('Nonfinite source cache')
  max_error=max(max_error,float(np.abs(a-MEAN).max()))
  if np.any(oe[t]!=0):raise ValueError('OE is not the documented all-zero placeholder; inspect before modifying metadata')
 if max_error>1.1:raise ValueError(f'S2 differs from documented constant mean placeholder: {max_error}')
 return {'source_missing_verified':True,'s2_max_abs_error_from_mean':max_error,'oe_all_zero':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('cache',type=Path);ap.add_argument('--apply-metadata',action='store_true');a=ap.parse_args()
 sdir=a.cache/'DynamicEarthNetS2/train';odir=a.cache/'DynamicEarthNetOE/train'
 result=verify(np.load(sdir/(RID+'.s2small.npy'),mmap_mode='r'),np.load(odir/(RID+'.oe32.npy'),mmap_mode='r'))
 marker=sdir/(RID+'.s2blk.npy.缺S2_用均值填充');result['marker_existed']=marker.is_file()
 if a.apply_metadata and not marker.exists():marker.write_text('S2 wholly missing; verified constant mean placeholder and all-zero OE. Metadata restored by source-only preflight 1110.\n',encoding='utf-8')
 result.update(marker=str(marker),marker_present=marker.is_file(),metadata_only=True,labels_modified=False)
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if not marker.is_file():raise SystemExit('Marker absent: rerun with --apply-metadata only after verifying the source-only report')
if __name__=='__main__':main()
