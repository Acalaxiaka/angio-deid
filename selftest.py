# -*- coding: utf-8 -*-
"""合成数据自测 (不含任何真实患者数据)."""
import os, sys, tempfile, shutil, numpy as np, cv2, pydicom, subprocess
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, JPEGBaseline8Bit, RLELossless
from pydicom.encaps import encapsulate
import deid_core as dc

FAKE = dict(name="ZHANG^SANFENG", pid="FAKEID778899", inst="TESTHOSPITAL_FAKE", phys="DR^LISIFAKE", acc="ACC55667788")

def frames_with_text(n=8, h=256, w=256, text=True):
    rng = np.random.default_rng(0); out = []
    for i in range(n):
        f = (rng.random((h, w)) * 60 + 80).astype(np.uint8)
        cv2.circle(f, (128 + i * 3, 128), 30, 30, 5)  # 移动的"血管"
        if text:
            cv2.putText(f, "ZHANG SANFENG", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
            cv2.putText(f, "ID FAKEID778899", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
            cv2.putText(f, "LAO 30 CRA 20", (150, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        out.append(f)
    return np.stack(out)

def make_xa(path, arr, ts=ExplicitVRLittleEndian, prim=30.0, sec=20.0, pid=FAKE["pid"], name=FAKE["name"], preamble=True):
    fm = FileMetaDataset(); fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"
    fm.MediaStorageSOPInstanceUID = generate_uid(); fm.TransferSyntaxUID = ts
    ds = Dataset(); ds.file_meta = fm
    ds.SOPClassUID = fm.MediaStorageSOPClassUID; ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.3.4.5.999"; ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = name; ds.PatientID = pid; ds.PatientBirthDate = "19600305"; ds.PatientSex = "M"; ds.PatientAge = "063Y"
    ds.PatientAddress = "FAKE ROAD 123"; ds.InstitutionName = FAKE["inst"]; ds.ReferringPhysicianName = FAKE["phys"]
    ds.PerformingPhysicianName = FAKE["phys"]; ds.AccessionNumber = FAKE["acc"]; ds.StudyDate = "20230615"; ds.StudyTime = "101010"
    ds.Modality = "XA"; ds.Manufacturer = "SIEMENS"; ds.ManufacturerModelName = "AXIOM-Artis"; ds.SeriesDescription = "Coro LCA"
    ds.ProtocolName = "Coronary"; ds.StationName = "CATHLAB1"
    ds.PositionerPrimaryAngle = prim; ds.PositionerSecondaryAngle = sec; ds.FrameTime = 66.7; ds.CineRate = 15
    ds.KVP = 80; ds.DistanceSourceToDetector = 1100; ds.ImagerPixelSpacing = [0.2, 0.2]
    ds.add_new(0x00191010, "LO", "SIEMENS PRIVATE " + FAKE["name"])
    ds.add_new(0x60003000, "OW", np.zeros(256 * 256 // 8, np.uint8).tobytes())
    n, h, w = arr.shape
    ds.Rows, ds.Columns, ds.NumberOfFrames = h, w, n
    ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"; ds.BitsAllocated = 8; ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0
    ds.FrameIncrementPointer = 0x00181063
    if ts == JPEGBaseline8Bit:
        ds.PixelData = encapsulate([cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes() for f in arr])
        ds["PixelData"].VR = "OB"
    elif ts == RLELossless:
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian; ds.PixelData = arr.tobytes()
        ds.compress(RLELossless)
    else:
        ds.PixelData = arr.tobytes()
    ds.save_as(path, enforce_file_format=preamble)

def make_video(path, arr):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe(); n, h, w = arr.shape
    codec = ["-c:v", "mjpeg", "-q:v", "3"] if path.endswith(".avi") else ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "%dx%d" % (w, h), "-r", "15", "-i", "-",
           "-metadata", "title=" + FAKE["name"], "-metadata", "comment=" + FAKE["pid"]] + codec + [path]
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
    subprocess.run(cmd, input=arr.tobytes(), check=True, **kw)

def run(report_path=None, keep=None):
    lines = []
    def L(s): lines.append(str(s))
    base = keep or tempfile.mkdtemp(prefix="angiodeid_test_")
    src = os.path.join(base, "in"); out = os.path.join(base, "out"); mp = os.path.join(base, "map")
    os.makedirs(os.path.join(src, "CD", "DICOM", "S1"), exist_ok=True)
    arr = frames_with_text()
    make_xa(os.path.join(src, "CD", "DICOM", "S1", "IM0001"), arr, prim=-30, sec=25)  # 无扩展名
    make_xa(os.path.join(src, "jpeg.dcm"), arr, ts=JPEGBaseline8Bit, prim=45, sec=-20)
    make_xa(os.path.join(src, "rle.dcm"), arr, ts=RLELossless, prim=2, sec=0, pid="OTHERFAKE001", name="LI^SIFAKE")
    open(os.path.join(src, "CD", "DICOMDIR"), "wb").write(b"\0" * 128 + b"DICM")
    open(os.path.join(src, "readme.txt"), "w").write("not dicom")
    rgb = np.stack([frames_with_text(text=True)] * 1)[0]
    make_video(os.path.join(src, "clip.avi"), rgb); make_video(os.path.join(src, "clip.mp4"), rgb)
    origs = {f: open(os.path.join(src, f), "rb").read() for f in ("jpeg.dcm", "rle.dcm")}
    ok = True
    for mode in ("auto", "corner", "none"):
        o = os.path.join(out, mode)
        s = dc.Session(o, mp, mask_mode=mode, log=L)
        rep = s.run([src])
        L("MODE %s outputs=%d problems=%d errors=%d" % (mode, rep["outputs"], len(rep["problems"]), rep["errors"]))
        if rep["outputs"] != 5 or rep["problems"] or rep["errors"]: ok = False
        for r in s.manifest_rows:
            L("  %s %s %s frames=%s prim=%s sec=%s proj=%s masked=%s" % (r["code"], r["modality"], r["age_group"], r["frames"], r["primary_angle"], r["secondary_angle"], r["projection"], r["masked_regions"]))
        # 角度检查
        dcms = [r for r in s.manifest_rows if r["modality"] == "XA"]
        projs = sorted(r["projection"] for r in dcms)
        if projs != sorted(["RAO 30 CRA 25", "LAO 45 CAU 20", "AP"]): ok = False; L("  投照角度错误 %s" % projs)
        if len({r["code"] for r in dcms}) != 2: ok = False; L("  患者编码数错误")
        # 遮挡检查: 左上角文字区像素
        for r in dcms:
            d = pydicom.dcmread(os.path.join(o, r["output_file"]))
            a = d.pixel_array
            tl = int(a[:, 5:30, 4:(60 if mode == "corner" else 120)].max())
            if mode != "none" and tl > 200: ok = False; L("  未遮挡文字 %s max=%d" % (r["output_file"], tl))
            if mode == "none" and tl < 200: ok = False
            if d.PatientIdentityRemoved != "YES" or "PatientBirthDate" in d or d.StudyDate == "20230615": ok = False; L("  标签问题")
            if float(d.PositionerPrimaryAngle) != float(r["primary_angle"]) or "KVP" not in d or "DistanceSourceToDetector" not in d: ok = False
        mpf = open(s.mapping_path, encoding="utf-8-sig").read()
        if FAKE["pid"] not in mpf: ok = False
        mnf = open(s.manifest_path, encoding="utf-8-sig").read()
        if FAKE["pid"] in mnf or "ZHANG" in mnf: ok = False; L("  manifest 含隐私!")
    for f, b in origs.items():
        if open(os.path.join(src, f), "rb").read() != b: ok = False; L("原文件被修改!")
    L("SELFTEST " + ("PASS" if ok else "FAIL"))
    txt = "\n".join(lines)
    if report_path:
        open(report_path, "w", encoding="utf-8").write(txt)
    if not keep: shutil.rmtree(base, ignore_errors=True)
    return ok, txt

if __name__ == "__main__":
    ok, txt = run(keep=sys.argv[1] if len(sys.argv) > 1 else None)
    print(txt); sys.exit(0 if ok else 1)
