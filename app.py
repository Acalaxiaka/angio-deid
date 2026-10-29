# -*- coding: utf-8 -*-
"""冠脉造影批量脱敏工具 - 图形界面 (离线运行)"""
import os, sys, threading, queue, datetime

def _selftest_mode():
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        rp = sys.argv[i + 1] if len(sys.argv) > i + 1 else os.path.join(os.getcwd(), "selftest_report.txt")
        try:
            import selftest
            ok, txt = selftest.run(report_path=rp)
        except Exception:
            import traceback
            open(rp, "w", encoding="utf-8").write("SELFTEST CRASH\n" + traceback.format_exc()); ok = False
        sys.exit(0 if ok else 1)
_selftest_mode()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import deid_core as dc
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except Exception:
    HAS_DND = False

MODES = [("自动检测文字区域", "auto"), ("四角黑框", "corner"), ("不遮挡", "none")]

class App:
    def __init__(self):
        self.root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
        r = self.root
        r.title("冠脉造影文件批量脱敏工具 v%s (离线)" % dc.VERSION); r.geometry("900x680")
        try: r.tk.call("tk", "scaling", 1.3)
        except Exception: pass
        self.q = queue.Queue(); self.cancel = threading.Event(); self.worker = None
        self.inputs = []
        top = ttk.LabelFrame(r, text="① 待处理文件（可多选，或把文件/文件夹直接拖进下方框内）")
        top.pack(fill="both", expand=False, padx=8, pady=4)
        bf = ttk.Frame(top); bf.pack(fill="x")
        ttk.Button(bf, text="选择文件", command=self.add_files).pack(side="left", padx=3, pady=3)
        ttk.Button(bf, text="选择文件夹", command=self.add_dir).pack(side="left", padx=3)
        ttk.Button(bf, text="清空列表", command=self.clear).pack(side="left", padx=3)
        self.cnt = ttk.Label(bf, text="已添加 0 项"); self.cnt.pack(side="left", padx=10)
        self.lb = tk.Listbox(top, height=7, selectmode="extended"); self.lb.pack(fill="both", expand=True, padx=3, pady=3)
        if HAS_DND:
            self.lb.drop_target_register(DND_FILES); self.lb.dnd_bind("<<Drop>>", self.on_drop)
            r.drop_target_register(DND_FILES); r.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.lb.insert("end", "（拖放不可用，请使用按钮）")
        of = ttk.LabelFrame(r, text="② 输出设置（原始文件不会被修改）"); of.pack(fill="x", padx=8, pady=4)
        self.out = tk.StringVar(); self.map = tk.StringVar()
        for i, (lab, var) in enumerate([("脱敏输出文件夹：", self.out), ("对照表保存文件夹（含隐私，须留院内）：", self.map)]):
            ttk.Label(of, text=lab).grid(row=i, column=0, sticky="w", padx=3, pady=2)
            ttk.Entry(of, textvariable=var, width=70).grid(row=i, column=1, sticky="we")
            ttk.Button(of, text="浏览…", command=lambda v=var: self.pick(v)).grid(row=i, column=2, padx=3)
        of.columnconfigure(1, weight=1)
        mf = ttk.LabelFrame(r, text="③ 图像内文字遮挡方式"); mf.pack(fill="x", padx=8, pady=4)
        self.mode = tk.StringVar(value="auto")
        for t, v in MODES: ttk.Radiobutton(mf, text=t, value=v, variable=self.mode).pack(side="left", padx=8)
        ttk.Label(mf, text="  四角框宽%:").pack(side="left"); self.fw = tk.IntVar(value=30)
        ttk.Spinbox(mf, from_=5, to=50, textvariable=self.fw, width=4).pack(side="left")
        ttk.Label(mf, text=" 高%:").pack(side="left"); self.fh = tk.IntVar(value=12)
        ttk.Spinbox(mf, from_=3, to=40, textvariable=self.fh, width=4).pack(side="left")
        rf = ttk.Frame(r); rf.pack(fill="x", padx=8, pady=4)
        self.start_btn = ttk.Button(rf, text="▶ 开始脱敏", command=self.start); self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(rf, text="■ 取消", command=self.do_cancel, state="disabled"); self.cancel_btn.pack(side="left", padx=5)
        ttk.Button(rf, text="打开输出文件夹", command=lambda: self.open_dir(self.out.get())).pack(side="left", padx=5)
        ttk.Button(rf, text="使用说明", command=self.help).pack(side="right")
        self.pb = ttk.Progressbar(r, mode="determinate"); self.pb.pack(fill="x", padx=8)
        self.st = ttk.Label(r, text="就绪"); self.st.pack(fill="x", padx=8)
        lf = ttk.LabelFrame(r, text="日志"); lf.pack(fill="both", expand=True, padx=8, pady=4)
        self.log = tk.Text(lf, height=12); sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set); sb.pack(side="right", fill="y"); self.log.pack(fill="both", expand=True)
        ttk.Label(r, foreground="#a00", text="声明：本工具仅辅助脱敏，结果须人工抽查；数据使用须遵守医院伦理及数据安全规定。对照表含隐私，严禁外传。").pack(fill="x", padx=8, pady=2)
        r.after(100, self.poll)

    def add(self, paths):
        for p in paths:
            p = os.path.normpath(p)
            if p and p not in self.inputs:
                self.inputs.append(p); self.lb.insert("end", p)
        self.cnt.config(text="已添加 %d 项" % len(self.inputs))
    def add_files(self): self.add(filedialog.askopenfilenames(title="选择文件（可多选）"))
    def add_dir(self):
        d = filedialog.askdirectory(title="选择文件夹（将递归扫描）")
        if d: self.add([d])
    def clear(self): self.inputs = []; self.lb.delete(0, "end"); self.cnt.config(text="已添加 0 项")
    def on_drop(self, e): self.add(self.root.tk.splitlist(e.data))
    def pick(self, v):
        d = filedialog.askdirectory()
        if d: v.set(os.path.normpath(d))
    def open_dir(self, d):
        if d and os.path.isdir(d) and os.name == "nt": os.startfile(d)
    def help(self):
        messagebox.showinfo("使用说明", "1. 点“选择文件/选择文件夹”或直接拖入（可多选，支持光盘 DICOM 目录、无扩展名文件、AVI/MP4）。\n"
            "2. 选择输出文件夹和对照表文件夹（建议两者分开，对照表留在院内电脑）。\n3. 选择遮挡方式，点“开始脱敏”。\n"
            "4. 完成后程序自动复查输出文件，并生成 manifest 清单（含投照角度标签，无隐私）。\n5. 请务必人工抽查若干图像，确认无残留姓名等文字。")
    def logw(self, s): self.q.put(("log", s))
    def prog(self, i, n, name): self.q.put(("prog", (i, n, name)))
    def start(self):
        if not self.inputs: return messagebox.showwarning("提示", "请先添加文件或文件夹")
        o, m = self.out.get().strip(), self.map.get().strip()
        if not o or not m: return messagebox.showwarning("提示", "请选择输出文件夹和对照表文件夹")
        for p in self.inputs:
            if os.path.isdir(p) and os.path.abspath(o).startswith(os.path.abspath(p) + os.sep):
                return messagebox.showwarning("提示", "输出文件夹不能位于输入文件夹内部")
        if not messagebox.askokcancel("确认", "对照表将保存到：\n%s\n\n该文件含患者真实信息，只能保存在医院内部，不得外传。继续？" % m): return
        self.cancel.clear(); self.start_btn.config(state="disabled"); self.cancel_btn.config(state="normal")
        s = dc.Session(o, m, self.mode.get(), self.fw.get() / 100.0, self.fh.get() / 100.0, log=self.logw, progress=self.prog, cancel=self.cancel)
        def job():
            try: rep = s.run(list(self.inputs)); self.q.put(("done", rep))
            except Exception as e: self.q.put(("log", "严重错误: %s" % e)); self.q.put(("done", None))
        self.worker = threading.Thread(target=job, daemon=True); self.worker.start()
    def do_cancel(self): self.cancel.set(); self.st.config(text="正在取消…")
    def poll(self):
        try:
            while True:
                k, v = self.q.get_nowait()
                if k == "log":
                    self.log.insert("end", "[%s] %s\n" % (datetime.datetime.now().strftime("%H:%M:%S"), v)); self.log.see("end")
                elif k == "prog":
                    i, n, name = v; self.pb["maximum"] = max(n, 1); self.pb["value"] = i; self.st.config(text="%d / %d  %s" % (i, n, name))
                elif k == "done":
                    self.start_btn.config(state="normal"); self.cancel_btn.config(state="disabled")
                    if v:
                        msg = "完成：输出 %d 个文件，失败 %d 个，复查问题 %d 处。\n请人工抽查图像！" % (v["outputs"], v["errors"], len(v["problems"]))
                        (messagebox.showwarning if v["problems"] or v["errors"] else messagebox.showinfo)("完成", msg)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

if __name__ == "__main__":
    App().root.mainloop()
