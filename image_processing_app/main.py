import tkinter as tk 
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

import operations
import pipeline
import saver

# Colors
BG = "#f0f0f0"
BTN_BG = "#4a90d9"
BTN_FG = "white"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.geometry("700x550")
        self.root.configure(bg=BG)

        self.original_img = None
        self.processed_img = None

        # Title
        tk.Label(root, text="Image Processing App", font=("Arial", 16, "bold"), bg=BG).pack(pady=10)

        # Images frame
        img_frame = tk.Frame(root, bg=BG)
        img_frame.pack(pady=10)

        # Original image
        left = tk.Frame(img_frame, bg=BG)
        left.pack(side="left", padx=20)
        tk.Label(left, text="Original", font=("Arial", 10, "bold"), bg=BG).pack()
        self.label1 = tk.Label(left, bg="#ddd")
        self.label1.pack()
        # Placeholder image
        self.placeholder1 = ImageTk.PhotoImage(Image.new("RGB", (300, 250), "#dddddd"))
        self.label1.config(image=self.placeholder1)

        # Processed image
        right = tk.Frame(img_frame, bg=BG)
        right.pack(side="left", padx=20)
        tk.Label(right, text="Processed", font=("Arial", 10, "bold"), bg=BG).pack()
        self.label2 = tk.Label(right, bg="#ddd")
        self.label2.pack()
        self.placeholder2 = ImageTk.PhotoImage(Image.new("RGB", (300, 250), "#dddddd"))
        self.label2.config(image=self.placeholder2)

        # Controls frame
        ctrl_frame = tk.Frame(root, bg=BG)
        ctrl_frame.pack(pady=15)

        # Upload button
        tk.Button(ctrl_frame, text="Upload Image", command=self.load_image,
                  bg=BTN_BG, fg=BTN_FG, font=("Arial", 10), width=14, pady=5).pack(side="left", padx=5)

        # Operation dropdown
        ops = [
            "histogram", "smooth", "sharpen",
            "gaussian_noise", "salt_pepper", "median",
            "grayscale", "color_enhance",
            "lowpass", "highpass"
        ]
        self.pipeline_ops = ops
        self.selected_steps = ["grayscale", "smooth", "lowpass"]

        self.operation = tk.StringVar()
        self.operation.set("histogram")

        tk.OptionMenu(ctrl_frame, self.operation, *ops).pack(side="left", padx=5)

        # Apply button
        tk.Button(ctrl_frame, text="Apply", command=self.apply_operation,
                  bg=BTN_BG, fg=BTN_FG, font=("Arial", 10), width=10, pady=5).pack(side="left", padx=5)

        # Save to Desktop button
        tk.Button(ctrl_frame, text="Save to Desktop", command=self.save_to_desktop,
                  bg="#28a745", fg=BTN_FG, font=("Arial", 10), width=14, pady=5).pack(side="left", padx=5)

        # Pipeline frame
        pipe_frame = tk.Frame(root, bg=BG)
        pipe_frame.pack(pady=10)

        self.pipeline_summary = tk.StringVar(value=f"Pipeline: {', '.join(self.selected_steps)}")
        tk.Label(pipe_frame, textvariable=self.pipeline_summary, font=("Arial", 9), bg=BG).pack()

        pipe_btns = tk.Frame(root, bg=BG)
        pipe_btns.pack(pady=5)

        tk.Button(pipe_btns, text="Choose Steps", command=self.open_pipeline_selector,
                  bg="#6c757d", fg=BTN_FG, font=("Arial", 10), width=12, pady=5).pack(side="left", padx=5)

        tk.Button(pipe_btns, text="Run Pipeline", command=self.run_pipeline,
                  bg=BTN_BG, fg=BTN_FG, font=("Arial", 10), width=12, pady=5).pack(side="left", padx=5)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
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
        if func is None:
            return

        self.processed_img = func(self.original_img)
        self.show_image(self.processed_img, self.label2)
        saver.save_image(self.processed_img, op)

    def run_pipeline(self):
        if self.original_img is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        if not (2 <= len(self.selected_steps) <= 4):
            messagebox.showerror("Selection Error", "Please select between 2 and 4 steps.")
            return

        self.processed_img = pipeline.apply_pipeline(self.original_img, self.selected_steps)
        self.show_image(self.processed_img, self.label2)
        saver.save_image(self.processed_img, "pipeline")

    def save_to_desktop(self):
        if self.processed_img is None:
            messagebox.showwarning("No Image", "No processed image to save.")
            return

        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        filename = filedialog.asksaveasfilename(
            initialdir=desktop,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")],
            initialfile="processed_image.png"
        )
        if filename:
            cv2.imwrite(filename, self.processed_img)
            messagebox.showinfo("Saved", f"Image saved to:\n{filename}")

    def open_pipeline_selector(self):
        window = tk.Toplevel(self.root)
        window.title("Select Pipeline Steps")
        window.geometry("250x380")
        window.configure(bg=BG)

        tk.Label(window, text="Choose 2-4 Steps", font=("Arial", 12, "bold"), bg=BG).pack(pady=15)

        selections = []
        for op_name in self.pipeline_ops:
            var = tk.IntVar(value=1 if op_name in self.selected_steps else 0)
            tk.Checkbutton(window, text=op_name, variable=var, bg=BG, 
                          font=("Arial", 10), anchor="w").pack(fill="x", padx=30)
            selections.append((op_name, var))

        def apply_selection():
            chosen = [name for name, var in selections if var.get() == 1]
            if not (2 <= len(chosen) <= 4):
                messagebox.showerror("Selection Error", "Please select between 2 and 4 steps.")
                return
            self.selected_steps = chosen
            self.pipeline_summary.set(f"Pipeline: {', '.join(chosen)}")
            window.destroy()

        tk.Button(window, text="Apply", command=apply_selection,
                  bg=BTN_BG, fg=BTN_FG, font=("Arial", 10), width=10, pady=5).pack(pady=20)

    def show_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((300, 250))
        img_tk = ImageTk.PhotoImage(img_pil)
        label.config(image=img_tk)
        label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
