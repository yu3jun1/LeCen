"""pylidc 轮廓 → 物理坐标 → 目标网格实例标签图。

pylidc 数据库只保存像素间距与各层 z 位置，没有 ImagePositionPatient 的 x、y 与 ImageOrientationPatient，
不足以完成体素 → 世界坐标转换，因此按 DICOM 方案读取原始 DICOM 几何（只需 888 个序列，路径由 ~/.pylidcrc 指定）。
pylidc 索引 (i, j, k) = (行, 列, 按 z 排序的层)；世界坐标 = IPP[k] + j·Δcol·row_cos + i·Δrow·col_cos。
目标网格上逐体素反查原始索引并取最近邻。
"""
import numpy as np
import pylidc as pl
import SimpleITK as sitk
from pylidc.utils import consensus

from common.coords import affine_of, vox_to_world, world_to_vox


class ScanGeometry:
    def __init__(self, scan):
        dcm = scan.load_all_dicom_images(verbose=False)
        self.ipp = np.array([d.ImagePositionPatient for d in dcm], dtype=float)
        iop = np.array(dcm[0].ImageOrientationPatient, dtype=float)
        self.row_cos, self.col_cos = iop[:3], iop[3:]
        self.d_row, self.d_col = map(float, dcm[0].PixelSpacing)
        self.normal = np.cross(self.row_cos, self.col_cos)
        self.slice_pos = self.ipp @ self.normal

    def to_world(self, ijk):
        i, j, k = ijk.T
        return self.ipp[k] + np.outer(j * self.d_col, self.row_cos) + np.outer(i * self.d_row, self.col_cos)

    def to_index(self, w):
        k = np.abs((w @ self.normal)[:, None] - self.slice_pos[None]).argmin(1)
        d = w - self.ipp[k]
        return np.stack([np.round(d @ self.col_cos / self.d_row), np.round(d @ self.row_cos / self.d_col), k], 1).astype(int)


def make_label_map(seriesuid, target_image):
    """返回目标网格上的实例标签图（uint16，值 = 簇序号 + 1）与各簇信息（共识 mask 质心的世界坐标）。"""
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == seriesuid).first()
    geo = ScanGeometry(scan)
    a = affine_of(target_image)
    shape = np.array(target_image.GetSize())[::-1]
    label = np.zeros(shape, dtype=np.uint16)
    clusters = []
    for n, anns in enumerate(scan.cluster_annotations(), 1):
        cmask, cbbox, _ = consensus(anns, clevel=0.5)
        lo = np.array([s.start for s in cbbox])
        world = geo.to_world(np.argwhere(cmask) + lo)
        tv = world_to_vox(world, a)
        t_lo = np.maximum(np.floor(tv.min(0)).astype(int) - 1, 0)
        t_hi = np.minimum(np.ceil(tv.max(0)).astype(int) + 2, shape)
        grid = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(t_lo, t_hi)], indexing="ij"), -1).reshape(-1, 3)
        src = geo.to_index(vox_to_world(grid.astype(float), a)) - lo
        inside = ((src >= 0) & (src < cmask.shape)).all(1)
        hit = np.zeros(len(grid), dtype=bool)
        hit[inside] = cmask[tuple(src[inside].T)]
        label[tuple(grid[hit].T)] = n
        clusters.append({"pylidc_cluster_id": n - 1, "label_value": n, "center_world_xyz": world.mean(0)})
    img = sitk.GetImageFromArray(label)
    img.CopyInformation(target_image)
    return img, clusters, scan.patient_id
