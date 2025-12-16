import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

import pipeline
import saver

IMG_SIZE = 320


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.geometry("820x520")
        self.root.resizable(False, False)

        self.original_img = None
        self.processed_img = None

        # =======================
        # MAIN LAYOUT
        # =======================
        main = tk.Frame(root, padx=12, pady=12)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main)
        left.grid(row=0, column=0, padx=10)

        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky="n")

        # =======================
        # IMAGE DISPLAY
        # =======================
        tk.Label(left, text="Original Image", font=("Arial", 11, "bold")).pack()
        self.label1 = tk.Label(left, bd=2, relief="groove", width=IMG_SIZE, height=IMG_SIZE)
        self.label1.pack(pady=(4, 12))

        tk.Label(left, text="Processed Image", font=("Arial", 11, "bold")).pack()
        self.label2 = tk.Label(left, bd=2, relief="groove", width=IMG_SIZE, height=IMG_SIZE)
        self.label2.pack()

        # =======================
        # CONTROLS
        # =======================
        tk.Label(right, text="Controls", font=("Arial", 12, "bold")).pack(anchor="w")

        tk.Button(
            right, text="Upload Image", width=22, command=self.load_image
        ).pack(pady=(8, 12))

        self.operation = tk.StringVar(value="histogram")

        ops = [
            "histogram", "smooth", "sharpen",
            "gaussian_noise", "salt_pepper", "median",
            "grayscale", "color_enhance",
            "lowpass", "highpass"
        ]

        tk.Label(right, text="Single Operation").pack(anchor="w")
        tk.OptionMenu(right, self.operation, *ops).pack(fill="x", pady=(2, 10))

        tk.Button(
            right, text="Apply Operation", width=22, command=self.apply_operation
        ).pack(pady=(0, 12))

        # =======================
        # PIPELINE
        # =======================
        tk.Label(right, text="Pipeline", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 0))

        self.pipeline_ops = ops
        self.selected_steps = ["grayscale", "smooth", "lowpass"]
        self.pipeline_summary = tk.StringVar()

        self.update_pipeline_text()

        tk.Button(
            right, text="Choose Pipeline Steps", width=22,
            command=self.open_pipeline_selector
        ).pack(pady=(6, 4))

        tk.Label(right, textvariable=self.pipeline_summary, wraplength=220, justify="left").pack()

        tk.Button(
            right, text="Run Pipeline", width=22, command=self.run_pipeline
        ).pack(pady=(10, 0))

    # =======================
    # LOGIC
    # =======================
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.bmp")]
        )
        if not path:
            return
        self.original_img = cv2.imread(path)
        self.show_image(self.original_img, self.label1)

    def apply_operation(self):
        if self.original_img is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        op = self.operation.get()
        func = pipeline.OPERATION_MAP.get(op)

        self.processed_img = func(self.original_img)
        self.show_image(self.processed_img, self.label2)
        saver.save_image(self.processed_img, op)

    def run_pipeline(self):
        if self.original_img is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        if not (2 <= len(self.selected_steps) <= 4):
            messagebox.showerror("Selection Error", "Select between 2 and 4 steps.")
            return

        self.processed_img = pipeline.apply_pipeline(
            self.original_img, self.selected_steps
        )
        self.show_image(self.processed_img, self.label2)
        saver.save_image(self.processed_img, "pipeline")

    # =======================
    # PIPELINE SELECTOR
    # =======================
    def open_pipeline_selector(self):
        window = tk.Toplevel(self.root)
        window.title("Select Pipeline Steps")
        window.geometry("260x360")

        vars_map = {}

        for op in self.pipeline_ops:
            var = tk.IntVar(value=1 if op in self.selected_steps else 0)
            tk.Checkbutton(window, text=op, variable=var, anchor="w").pack(fill="x", padx=8)
            vars_map[op] = var

        def apply():
            chosen = [k for k, v in vars_map.items() if v.get()]
            if not (2 <= len(chosen) <= 4):
                messagebox.showerror("Error", "Select between 2 and 4 steps.")
                return
            self.selected_steps = chosen
            self.update_pipeline_text()
            window.destroy()

        tk.Button(window, text="Apply Selection", command=apply).pack(pady=10)

    # =======================
    # HELPERS
    # =======================
    def update_pipeline_text(self):
        self.pipeline_summary.set("Selected:\n• " + "\n• ".join(self.selected_steps))

    def show_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((IMG_SIZE, IMG_SIZE))
        img_tk = ImageTk.PhotoImage(img_pil)
        label.config(image=img_tk)
        label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
