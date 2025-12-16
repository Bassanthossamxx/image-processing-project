import tkinter as tk 
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

import operations
import pipeline
import saver

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.original_img = None
        self.processed_img = None

        self.label1 = tk.Label(root)
        self.label1.pack()

        self.label2 = tk.Label(root)
        self.label2.pack()

        tk.Button(root, text="Upload Image", command=self.load_image).pack()

        self.operation = tk.StringVar()
        self.operation.set("histogram")

        ops = [
            "histogram", "smooth", "sharpen",
            "gaussian_noise", "salt_pepper", "median",
            "grayscale", "color_enhance",
            "lowpass", "highpass"
        ]

        self.pipeline_ops = ops
        self.selected_steps = ["grayscale", "smooth", "lowpass"]
        self.pipeline_summary = tk.StringVar(value=f"Selected: {', '.join(self.selected_steps)}")

        tk.OptionMenu(root, self.operation, *ops).pack()

        tk.Button(root, text="Apply Operation", command=self.apply_operation).pack()
        tk.Button(root, text="Choose Pipeline Steps", command=self.open_pipeline_selector).pack()
        tk.Label(root, textvariable=self.pipeline_summary).pack()
        tk.Button(root, text="Run Pipeline", command=self.run_pipeline).pack()

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.original_img = cv2.imread(path)
        self.show_image(self.original_img, self.label1)

    def apply_operation(self):
        if self.original_img is None:
            return

        op = self.operation.get()
        func = pipeline.OPERATION_MAP.get(op)
        if func is None:
            return

        self.processed_img = func(self.original_img)
        self.show_image(self.processed_img, self.label2)
        saver.save_image(self.processed_img, op)

    def run_pipeline(self):
        if self.original_img is None:
            return

        if not (2 <= len(self.selected_steps) <= 4):
            messagebox.showerror("Selection Error", "Please select between 2 and 4 steps.")
            return

        self.processed_img = pipeline.apply_pipeline(self.original_img, self.selected_steps)
        self.show_image(self.processed_img, self.label2)
        saver.save_image(self.processed_img, "pipeline")

    def open_pipeline_selector(self):
        window = tk.Toplevel(self.root)
        window.title("Select pipeline steps")

        selections = []
        for op_name in self.pipeline_ops:
            var = tk.IntVar(value=1 if op_name in self.selected_steps else 0)
            tk.Checkbutton(window, text=op_name, variable=var, anchor="w").pack(fill="x")
            selections.append((op_name, var))

        def apply_selection():
            chosen = [name for name, var in selections if var.get() == 1]
            if not (2 <= len(chosen) <= 4):
                messagebox.showerror("Selection Error", "Please select between 2 and 4 steps.")
                return
            self.selected_steps = chosen
            self.pipeline_summary.set(f"Selected: {', '.join(chosen)}")
            window.destroy()

        tk.Button(window, text="Apply", command=apply_selection).pack(pady=(8, 0))

    def show_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_pil)
        label.config(image=img_tk)
        label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
