# -*- coding: utf-8 -*-
"""冠脉造影文件批量脱敏核心模块 (离线). DICOM (XA 多帧/压缩) + AVI/MP4 视频."""
import os, sys, csv, re, hashlib, random, secrets, datetime, subprocess, tempfile, traceback
import numpy as np
import cv2
import pydicom
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.dataset import Dataset

VERSION = "1.0.0"
VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v", ".flv"}
DICOMDIR_CLASS = "1.2.840.10008.1.3.10"

# ---------- 标签规则 (参考 DICOM PS3.15 Annex E Basic Profile) ----------
# 需要整体删除的标签 / 序列
REMOVE_KEYWORDS = {
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames", "OtherPatientIDsSequence",
    "PatientMotherBirthName", "MilitaryRank", "BranchOfService", "MedicalRecordLocator",
    "EthnicGroup", "Occupation", "AdditionalPatientHistory", "PatientComments", "PatientReligiousPreference",
    "PatientBirthName", "PatientInsurancePlanCodeSequence", "IssuerOfPatientID", "PatientSize", "PatientWeight",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName", "InstitutionCodeSequence",
    "ReferringPhysicianName", "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName", "NameOfPhysiciansReadingStudy", "OperatorsName", "PhysiciansOfRecord",
    "RequestingPhysician", "ScheduledPerformingPhysicianName", "AccessionNumber", "StudyID",
    "StationName", "DeviceSerialNumber", "PlateID", "RequestAttributesSequence", "ReferencedPatientSequence",
    "ReferencedStudySequence", "ReferencedPerformedProcedureStepSequence", "PerformedProcedureStepID",
    "PerformedProcedureStepDescription", "RequestedProcedureID", "RequestedProcedureDescription",
    "ScheduledProcedureStepID", "AdmissionID", "AdmittingDiagnosesDescription", "ImageComments",
    "StudyComments", "InterpretationText", "TextComments", "OtherPatientIDsSequence", "PersonName",
    "ContentCreatorName", "IconImageSequence", "DerivationDescription", "StudyDescription",
    "PatientState", "CurrentPatientLocation", "PatientInstitutionResidence", "RegionOfResidence",
    "CountryOfResidence", "ResponsiblePerson", "ResponsibleOrganization", "Allergies",
    "MedicalAlerts", "PregnancyStatus", "SmokingStatus", "LastMenstrualDate", "InstanceCreatorUID",
}
# 保留的文本类标签 (其余 LO/SH/ST/LT/UT/UC/AE 文本一律删除, 防止漏网 PHI)
KEEP_TEXT = {
    "Modality", "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "SeriesDescription",
    "ProtocolName", "BodyPartExamined", "PatientSex", "PatientAge", "ImageType", "ConversionType",
    "PhotometricInterpretation", "PositionerMotion", "PositionerType", "ViewPosition",
    "FrameIncrementPointer", "PatientPosition", "Laterality", "ImageLaterality", "PatientOrientation",
    "LossyImageCompression", "LossyImageCompressionMethod", "BurnedInAnnotation", "PixelIntensityRelationship",
    "PatientIdentityRemoved", "DeidentificationMethod", "SpecificCharacterSet", "RescaleType",
    "ShutterShape", "FrameLabel", "CodeValue", "CodingSchemeDesignator", "CodeMeaning",
    "AcquisitionDeviceProcessingDescription", "Grid", "FilterType", "ExposureControlMode",
    "TypeOfDetectorMotion", "RecommendedViewingMode", "ImagerPixelSpacing", "ContentQualification",
}
# 不重映射的 UID (类 UID / 传输语法)
KEEP_UID = {"SOPClassUID", "MediaStorageSOPClassUID", "TransferSyntaxUID", "ReferencedSOPClassUID",
            "ImplementationClassUID", "RelatedGeneralSOPClassUID", "OriginalSpecializedSOPClassUID",
            "CodingSchemeUID", "MappingResourceUID", "ContextGroupExtensionCreatorUID"}
TEXT_VRS = {"LO", "SH", "ST", "LT", "UT", "UC", "AE"}

def log_noop(*a, **k): pass

# ---------- 文件扫描 ----------
def is_dicom(path):
    try:
        with open(path, "rb") as f:
            h = f.read(132)
        if len(h) == 132 and h[128:132] == b"DICM":
            return True
        if len(h) < 8:
            return False
        # 无前导码的老文件: 第一个标签通常为 (0008,xxxx) 组
        if h[0:2] in (b"\x08\x00", b"\x02\x00"):
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            return "SOPClassUID" in ds or "PatientID" in ds or "Modality" in ds
    except Exception:
        return False
    return False

def scan_inputs(paths, cancel=None):
    """返回 [(path, 'dicom'|'video'|'dicomdir')]"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in sorted(names):
                    files.append(os.path.join(root, n))
        elif os.path.isfile(p):
            files.append(p)
    out, seen = [], set()
    for f in files:
        if cancel and cancel.is_set():
            break
        rp = os.path.abspath(f)
        if rp in seen:
            continue
        seen.add(rp)
        ext = os.path.splitext(f)[1].lower()
        if os.path.basename(f).upper() == "DICOMDIR":
            out.append((rp, "dicomdir")); continue
        if ext in VIDEO_EXT:
            out.append((rp, "video")); continue
        if is_dicom(rp):
            out.append((rp, "dicom"))
    return out

# ---------- 遮挡 ----------
def corner_boxes(h, w, frac_w=0.25, frac_h=0.12):
    bw, bh = int(w * frac_w), int(h * frac_h)
    return [(0, 0, bw, bh), (w - bw, 0, w, bh), (0, h - bh, bw, h), (w - bw, h - bh, w, h)]

def _to_gray8(fr):
    a = fr
    if a.ndim == 3:
        a = a.mean(axis=2) if a.shape[2] != 1 else a[..., 0]
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return ((a - lo) * 255.0 / (hi - lo)).astype(np.uint8)

def detect_text_boxes(frames, band=0.22, side=0.30):
    """无 OCR 启发式: 在四周边带中寻找跨帧静止、高对比、笔画状的区域."""
    n = len(frames)
    idx = np.linspace(0, n - 1, min(n, 12)).astype(int)
    g = np.stack([_to_gray8(frames[i]) for i in idx]).astype(np.float32)
    h, w = g.shape[1:]
    mean = g.mean(0); std = g.std(0)
    m8 = mean.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(m8, cv2.MORPH_GRADIENT, k).astype(np.float32)
    tophat = np.maximum(cv2.morphologyEx(m8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))),
                        cv2.morphologyEx(m8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))))
    static = std < (6 if n > 1 else 255)
    cand = static & (grad > 60) & (tophat > 50)
    region = np.zeros((h, w), bool)
    bh, sw = int(h * band), int(w * side)
    region[:bh, :] = True; region[h - bh:, :] = True
    region[:, :sw] |= True; region[:, w - sw:] |= True
    # 中间部分只保留上/下边带
    region[bh:h - bh, sw:w - sw] = False
    cand &= region
    m = cand.astype(np.uint8) * 255
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 60), max(5, h // 120))))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    boxes = []
    for i in range(1, num):
        x, y, bw_, bh_, area = stats[i]
        if area < 40 or bh_ > h * 0.25 or bw_ > w * 0.9:
            continue
        pad = 4
        boxes.append((max(0, x - pad), max(0, y - pad), min(w, x + bw_ + pad), min(h, y + bh_ + pad)))
    return boxes

def compute_boxes(frames, mode, frac_w=0.25, frac_h=0.12):
    h, w = frames[0].shape[:2]
    if mode == "none":
        return []
    if mode == "corner":
        return corner_boxes(h, w, frac_w, frac_h)
    return detect_text_boxes(frames)

def apply_boxes(arr, boxes, multi):
    if not boxes:
        return arr
    arr = arr.copy()
    for (x0, y0, x1, y1) in boxes:
        if multi:
            arr[:, y0:y1, x0:x1, ...] = 0
        else:
            arr[y0:y1, x0:x1, ...] = 0
    return arr

# ---------- 工具 ----------
def projection_label(prim, sec, thr=5.0):
    try:
        p = float(prim)
    except Exception:
        return ""
    lr = "AP" if abs(p) <= thr else ("LAO %d" % round(p) if p > 0 else "RAO %d" % round(-p))
    try:
        s = float(sec)
        cc = "" if abs(s) <= thr else ("CRA %d" % round(s) if s > 0 else "CAU %d" % round(-s))
    except Exception:
        cc = ""
    return (lr + " " + cc).strip()

def age_group(age_str, birth, study):
    yrs = None
    m = re.match(r"^(\d{1,3})([YMWD]?)$", (age_str or "").strip().upper())
    if m:
        v = int(m.group(1)); u = m.group(2) or "Y"
        yrs = v if u == "Y" else v / 12 if u == "M" else 0
    elif birth and study:
        try:
            b = datetime.datetime.strptime(str(birth)[:8], "%Y%m%d"); s = datetime.datetime.strptime(str(study)[:8], "%Y%m%d")
            yrs = (s - b).days // 365
        except Exception:
            pass
    if yrs is None:
        return "", ""
    yrs = int(yrs)
    if yrs >= 90:
        return "090Y", "90+"
    lo = (yrs // 10) * 10
    return "%03dY" % yrs, "%d-%d" % (lo, lo + 9)

def shift_da(v, days):
    try:
        d = datetime.datetime.strptime(str(v)[:8], "%Y%m%d") + datetime.timedelta(days=days)
        return d.strftime("%Y%m%d") + str(v)[8:] if len(str(v)) > 8 else d.strftime("%Y%m%d")
    except Exception:
        return ""

# ---------- 会话 ----------
class Session:
    def __init__(self, out_dir, map_dir, mask_mode="auto", frac_w=0.25, frac_h=0.12,
                 log=log_noop, progress=log_noop, cancel=None, prefix="P"):
        self.out_dir, self.map_dir = out_dir, map_dir
        self.mask_mode, self.frac_w, self.frac_h = mask_mode, frac_w, frac_h
        self.log, self.progress, self.cancel = log, progress, cancel
        self.prefix = prefix
        self.salt = secrets.token_hex(16)
        self.patients = {}      # key -> dict(code, offset, studies{uid:idx}, series{uid:idx})
        self.uid_map = {}
        self.mapping_rows = []
        self.manifest_rows = []
        self.phi_strings = set()
        self.video_count = 0
        self.errors = 0
        self.outputs = []

    def new_uid(self, old):
        if old not in self.uid_map:
            h = int(hashlib.sha256((self.salt + str(old)).encode()).hexdigest()[:30], 16)
            self.uid_map[old] = "2.25." + str(h)
        return self.uid_map[old]

    def patient(self, pid, pname, sex):
        key = (pid or "") + "|" + (pname or "")
        if key not in self.patients:
            code = "%s%04d" % (self.prefix, len(self.patients) + 1)
            self.patients[key] = dict(code=code, offset=-random.randint(30, 3650),
                                      studies={}, series={})
            self.mapping_rows.append(dict(code=code, original_patient_id=pid, original_patient_name=pname, sex=sex))
        return self.patients[key]

    # ---- DICOM ----
    def _collect_phi(self, ds):
        for kw in ("PatientName", "PatientID", "PatientBirthDate", "InstitutionName", "InstitutionAddress",
                   "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName", "AccessionNumber",
                   "PatientAddress", "OtherPatientIDs", "StationName", "PatientTelephoneNumbers"):
            v = ds.get(kw)
            if v is None: continue
            vals = v if isinstance(v, (list, pydicom.multival.MultiValue)) else [v]
            for x in vals:
                s = str(x).strip()
                if len(s) >= 4 and s.upper() not in ("ANONYMOUS", "NONE", "UNKNOWN"):
                    self.phi_strings.add(s)
                    for part in re.split(r"[\^ ]", s):
                        if len(part) >= 4: self.phi_strings.add(part)

    def _clean(self, ds, offset):
        for elem in list(ds):
            tag = elem.tag
            kw = elem.keyword
            # 私有标签、曲线(50xx)、覆盖层(60xx, 可能含文字) 全部删除
            if tag.is_private or (tag.group & 0xFF00) in (0x5000, 0x6000):
                del ds[tag]; continue
            if kw in REMOVE_KEYWORDS:
                del ds[tag]; continue
            vr = elem.VR
            if vr == "SQ":
                for item in elem.value:
                    self._clean(item, offset)
                continue
            if tag == 0x7FE00010:
                continue
            if vr == "PN":
                del ds[tag]; continue
            if vr in ("DA", "DT"):
                if elem.value:
                    if elem.VM > 1:
                        elem.value = [shift_da(v, offset) for v in elem.value]
                    else:
                        elem.value = shift_da(elem.value, offset)
                continue
            if vr == "UI":
                if kw in KEEP_UID or not elem.value:
                    continue
                if elem.VM > 1:
                    elem.value = [self.new_uid(v) for v in elem.value]
                else:
                    elem.value = self.new_uid(elem.value)
                continue
            if vr in TEXT_VRS and kw not in KEEP_TEXT:
                del ds[tag]; continue
            if vr == "UN":
                del ds[tag]; continue

    def process_dicom(self, path):
        ds = pydicom.dcmread(path, force=True)
        fm = getattr(ds, "file_meta", None)
        if str(ds.get("SOPClassUID", "")) == DICOMDIR_CLASS or (fm is not None and str(fm.get("MediaStorageSOPClassUID", "")) == DICOMDIR_CLASS) or "DirectoryRecordSequence" in ds:
            self.log("跳过 DICOMDIR 目录文件: %s" % os.path.basename(path)); return
        self._collect_phi(ds)
        pid, pname = str(ds.get("PatientID", "")), str(ds.get("PatientName", ""))
        sex = str(ds.get("PatientSex", ""))
        age, agrp = age_group(str(ds.get("PatientAge", "")), ds.get("PatientBirthDate"), ds.get("StudyDate") or ds.get("SeriesDate"))
        p = self.patient(pid, pname, sex)
        st_uid, se_uid = str(ds.get("StudyInstanceUID", "nostudy")), str(ds.get("SeriesInstanceUID", "noseries"))
        st = p["studies"].setdefault(st_uid, len(p["studies"]) + 1)
        se = p["series"].setdefault(se_uid, len(p["series"]) + 1)
        nframes = int(ds.get("NumberOfFrames", 1) or 1)
        masked_boxes = []
        if "PixelData" in ds:
            ts = ds.file_meta.get("TransferSyntaxUID") if hasattr(ds, "file_meta") else None
            if ts is None:
                ds.file_meta = getattr(ds, "file_meta", pydicom.dataset.FileMetaDataset())
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                ts = ds.file_meta.TransferSyntaxUID
            arr = ds.pixel_array
            multi = nframes > 1
            frames = arr if multi else arr[None]
            if self.mask_mode != "none":
                masked_boxes = compute_boxes(frames, self.mask_mode, self.frac_w, self.frac_h)
                arr = apply_boxes(arr, masked_boxes, multi)
            was_compressed = pydicom.uid.UID(ts).is_compressed
            if was_compressed or masked_boxes:
                if was_compressed:
                    if arr.ndim == (4 if multi else 3) and str(ds.get("PhotometricInterpretation", "")).startswith("YBR"):
                        ds.PhotometricInterpretation = "RGB"
                    if "PlanarConfiguration" in ds: ds.PlanarConfiguration = 0
                    if ts in (pydicom.uid.JPEGBaseline8Bit, pydicom.uid.JPEGExtended12Bit) or "JPEG2000" in str(pydicom.uid.UID(ts).name) and "Lossless" not in str(pydicom.uid.UID(ts).name):
                        ds.LossyImageCompression = "01"
                arr = np.ascontiguousarray(arr)
                if arr.dtype.itemsize != (int(ds.BitsAllocated) // 8 or 1):
                    arr = arr.astype(np.uint16 if int(ds.BitsAllocated) == 16 else np.uint8)
                ds.PixelData = arr.tobytes()
                ds["PixelData"].VR = "OW" if int(ds.BitsAllocated) > 8 else "OB"
                ds["PixelData"].is_undefined_length = False
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        if self.mask_mode != "none":
            ds.BurnedInAnnotation = "NO" if masked_boxes else ds.get("BurnedInAnnotation", "")
        self._clean(ds, p["offset"])
        if hasattr(ds, "file_meta"):
            for kw in ("SourceApplicationEntityTitle", "SendingApplicationEntityTitle", "PrivateInformation",
                       "PrivateInformationCreatorUID", "ReceivingApplicationEntityTitle"):
                if kw in ds.file_meta: del ds.file_meta[kw]
            if "SOPInstanceUID" in ds: ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            if "SOPClassUID" in ds: ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            ds.file_meta.ImplementationVersionName = "ANGIODEID_" + VERSION.replace(".", "")
        ds.PatientName = p["code"]; ds.PatientID = p["code"]
        if age: ds.PatientAge = age
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = ("angio-deid %s PS3.15E basic;dates shifted;mask=%s" % (VERSION, self.mask_mode))[:64]
        cs = Dataset(); cs.CodeValue = "113100"; cs.CodingSchemeDesignator = "DCM"; cs.CodeMeaning = "Basic Application Confidentiality Profile"
        cs2 = Dataset(); cs2.CodeValue = "113107"; cs2.CodingSchemeDesignator = "DCM"; cs2.CodeMeaning = "Retain Longitudinal Temporal Information Modified Dates Option"
        ds.DeidentificationMethodCodeSequence = [cs, cs2]
        ds.SpecificCharacterSet = "ISO_IR 192"
        rel = os.path.join(p["code"], "STUDY%02d" % st, "SERIES%03d" % se)
        os.makedirs(os.path.join(self.out_dir, rel), exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(self.out_dir, rel, "IMG%04d.dcm" % i)): i += 1
        relf = os.path.join(rel, "IMG%04d.dcm" % i)
        ds.save_as(os.path.join(self.out_dir, relf), enforce_file_format=True)
        self.outputs.append(os.path.join(self.out_dir, relf))
        prim, secd = ds.get("PositionerPrimaryAngle", ""), ds.get("PositionerSecondaryAngle", "")
        ft = ds.get("FrameTime", ""); cr = ds.get("CineRate", "") or ds.get("RecommendedDisplayFrameRate", "")
        self.manifest_rows.append(dict(code=p["code"], sex=sex, age_group=agrp, modality=str(ds.get("Modality", "")),
            frames=nframes, rows=ds.get("Rows", ""), cols=ds.get("Columns", ""),
            primary_angle=prim, secondary_angle=secd, projection=projection_label(prim, secd),
            frame_time_ms=ft, cine_rate=cr, series_description=str(ds.get("SeriesDescription", "")),
            masked_regions=len(masked_boxes), output_file=relf.replace("\\", "/")))
        for r in self.mapping_rows:
            if r["code"] == p["code"]:
                r.setdefault("sources", []).append(path)

    # ---- 视频 ----
    def process_video(self, path):
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cap = cv2.VideoCapture(path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        ok = cap.isOpened()
        if ok:
            while True:
                r, fr = cap.read()
                if not r: break
                frames.append(fr)
                if self.cancel and self.cancel.is_set(): return
        cap.release()
        if not frames:  # 回退到 ffmpeg 解码
            rd = imageio_ffmpeg.read_frames(path, output_params=[], pix_fmt="bgr24")
            meta = rd.__next__(); w, h = meta["size"]; fps = meta.get("fps", 15)
            for b in rd:
                frames.append(np.frombuffer(b, np.uint8).reshape(h, w, 3))
        if not frames:
            raise RuntimeError("无法解码视频")
        if not fps or fps > 240: fps = 15
        arr = np.stack(frames)
        boxes = compute_boxes(arr, self.mask_mode, self.frac_w, self.frac_h) if self.mask_mode != "none" else []
        arr = apply_boxes(arr, boxes, True)
        n, h, w = arr.shape[:3]
        # H.264 需要偶数尺寸
        h2, w2 = h - h % 2, w - w % 2
        arr = np.ascontiguousarray(arr[:, :h2, :w2])
        self.video_count += 1
        code = "V%04d" % self.video_count
        os.makedirs(os.path.join(self.out_dir, "VIDEOS"), exist_ok=True)
        relf = "VIDEOS/%s.mp4" % code
        outp = os.path.join(self.out_dir, relf)
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w2, h2),
               "-r", "%.3f" % fps, "-i", "-", "-map_metadata", "-1", "-map_chapters", "-1", "-an", "-sn",
               "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
               "-metadata", "title=", "-metadata", "comment=deidentified", "-fflags", "+bitexact",
               "-flags:v", "+bitexact", "-movflags", "+faststart", outp]
        kw = {}
        if os.name == "nt": kw["creationflags"] = 0x08000000
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
        _, err = pr.communicate(arr.tobytes())
        if pr.returncode != 0:
            raise RuntimeError("ffmpeg 编码失败: " + err.decode("utf-8", "ignore")[-300:])
        self.outputs.append(outp)
        self.mapping_rows.append(dict(code=code, original_patient_id="", original_patient_name="(视频，未知患者)", sex="", sources=[path]))
        self.manifest_rows.append(dict(code=code, sex="", age_group="", modality="VIDEO", frames=n, rows=h2, cols=w2,
            primary_angle="", secondary_angle="", projection="", frame_time_ms="%.2f" % (1000.0 / fps), cine_rate="%.2f" % fps,
            series_description="", masked_regions=len(boxes), output_file=relf))

    # ---- 主流程 ----
    def run(self, inputs):
        os.makedirs(self.out_dir, exist_ok=True); os.makedirs(self.map_dir, exist_ok=True)
        if os.path.abspath(self.out_dir) == os.path.abspath(self.map_dir):
            self.log("警告: 对照表与输出在同一文件夹, 建议分开存放!")
        self.log("正在扫描文件 ...")
        items = scan_inputs(inputs, self.cancel)
        nd = sum(1 for _, t in items if t == "dicom"); nv = sum(1 for _, t in items if t == "video")
        self.log("找到 DICOM %d 个, 视频 %d 个, DICOMDIR %d 个(跳过)" % (nd, nv, sum(1 for _, t in items if t == "dicomdir")))
        work = [x for x in items if x[1] in ("dicom", "video")]
        for i, (pth, typ) in enumerate(work):
            if self.cancel and self.cancel.is_set():
                self.log("已取消."); break
            self.progress(i, len(work), os.path.basename(pth))
            try:
                if typ == "dicom": self.process_dicom(pth)
                else: self.process_video(pth)
                self.log("✓ %s" % os.path.basename(pth))
            except Exception as e:
                self.errors += 1
                self.log("✗ 失败(已跳过) %s : %s" % (os.path.basename(pth), e))
        self.progress(len(work), len(work), "完成")
        self.write_tables()
        rep = self.verify()
        return rep

    def write_tables(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mp = os.path.join(self.map_dir, "对照表_MAPPING_含隐私_勿外传_%s.csv" % ts)
        with open(mp, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f); wr.writerow(["code", "original_patient_id", "original_patient_name", "sex", "source_files"])
            for r in self.mapping_rows:
                wr.writerow([r["code"], r["original_patient_id"], r["original_patient_name"], r.get("sex", ""), " | ".join(r.get("sources", []))])
        self.mapping_path = mp
        mf = os.path.join(self.out_dir, "manifest_%s.csv" % ts)
        cols = ["code", "sex", "age_group", "modality", "frames", "rows", "cols", "primary_angle", "secondary_angle",
                "projection", "frame_time_ms", "cine_rate", "series_description", "masked_regions", "output_file"]
        with open(mf, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
            for r in self.manifest_rows: wr.writerow(r)
        self.manifest_path = mf
        self.log("对照表(含隐私, 请留在院内): %s" % mp)
        self.log("清单 manifest(无隐私): %s" % mf)

    def verify(self):
        self.log("正在复查输出文件 ...")
        problems = []
        phi = [s for s in self.phi_strings if len(s) >= 4]
        phib = [s.encode("utf-8", "ignore") for s in phi] + [s.encode("gbk", "ignore") for s in phi]
        phib = [b for b in set(phib) if len(b) >= 4]
        for f in self.outputs:
            rel = os.path.relpath(f, self.out_dir)
            try:
                if f.lower().endswith(".dcm"):
                    ds = pydicom.dcmread(f)
                    for e in ds.iterall():
                        if e.tag.is_private: problems.append("%s: 残留私有标签 %s" % (rel, e.tag))
                        if e.VR == "PN" and e.keyword != "PatientName" and str(e.value).strip(): problems.append("%s: 残留人名 %s" % (rel, e.keyword))
                        if e.keyword in ("PatientBirthDate", "InstitutionName", "AccessionNumber", "PatientAddress", "ReferringPhysicianName"):
                            problems.append("%s: 残留 %s" % (rel, e.keyword))
                    if ds.get("PatientIdentityRemoved") != "YES": problems.append("%s: 未标记 PatientIdentityRemoved" % rel)
                    data = open(f, "rb").read()
                    if "PixelData" in ds:
                        # 只搜头部(像素外)
                        data = data.replace(ds.PixelData, b"")
                else:
                    data = open(f, "rb").read()
                for b in phib:
                    if b in data: problems.append("%s: 发现原始隐私字符串 '%s'" % (rel, b.decode("utf-8", "ignore")))
            except Exception as e:
                problems.append("%s: 复查失败 %s" % (rel, e))
        if problems:
            for p in problems[:200]: self.log("⚠ " + p)
        self.log("复查完成: %d 个输出文件, 发现问题 %d 处, 处理失败 %d 个." % (len(self.outputs), len(problems), self.errors))
        return dict(outputs=len(self.outputs), problems=problems, errors=self.errors,
                    mapping=getattr(self, "mapping_path", ""), manifest=getattr(self, "manifest_path", ""))
