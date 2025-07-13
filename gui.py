"""
Image Colorization GUI

This module provides a graphical user interface for the image colorization system.
It includes features like drag and drop, real-time feedback, and quality metrics.

Author: Siddh
Date: December 2024
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import os
import threading
import time
from pathlib import Path
import logging
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
from .colorizer import ImageColorizer
except ImportError:
    from colorizer import ImageColorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernColorizationGUI:
    """
    GUI for Image Colorization
    
    This class provides a graphical user interface for the image colorization system.
    Features include drag and drop, real-time feedback, and quality metrics.
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI.
        
        Args:
            root (tk.Tk): Root Tkinter window
        """
        self.root = root
        self.root.title("Advanced Image Colorization System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize colorizer
        try:
            self.colorizer = ImageColorizer()
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_loaded = False
        
        # Variables
        self.original_image = None
        self.colorized_image = None
        self.original_path = None
        self.processing = False
        
        # Setup GUI
        self._setup_styles()
        self._create_widgets()
        self._setup_layout()
        
        logger.info("GUI initialized successfully")
    
    def _setup_styles(self):
        """Configure modern styling for the GUI."""
        style = ttk.Style()
        style.theme_use('clam')  # Using clam theme - looks pretty good
        
        # Configure colors - trying to make it look nice
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        
        # Configure buttons - different colors for different actions
        style.configure('Primary.TButton', 
                      font=('Arial', 10, 'bold'),
                      background='#3498db',
                      foreground='white')
        style.configure('Success.TButton',
                      font=('Arial', 10, 'bold'),
                      background='#27ae60',
                      foreground='white')
        style.configure('Warning.TButton',
                      font=('Arial', 10, 'bold'),
                      background='#f39c12',
                      foreground='white')
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # Title
        self.title_label = ttk.Label(
            self.main_frame,
            text="Advanced Image Colorization System",
            style='Title.TLabel'
        )
        
        # Control panel
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="10")
        
        # File selection
        self.file_frame = ttk.Frame(self.control_frame)
        self.file_label = ttk.Label(self.file_frame, text="Select Image:", style='Header.TLabel')
        self.file_button = ttk.Button(
            self.file_frame,
            text="Browse",
            command=self._browse_file,
            style='Primary.TButton'
        )
        self.file_path_var = tk.StringVar(value="No file selected")
        self.file_path_label = ttk.Label(self.file_frame, textvariable=self.file_path_var, style='Info.TLabel')
        
        # Processing buttons
        self.process_frame = ttk.Frame(self.control_frame)
        self.process_button = ttk.Button(
            self.process_frame,
            text="Colorize Image",
            command=self._process_image,
            style='Success.TButton',
            state='disabled'
        )
        self.batch_button = ttk.Button(
            self.process_frame,
            text="Batch Process",
            command=self._batch_process,
            style='Warning.TButton'
        )
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.control_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_label = ttk.Label(self.control_frame, text="", style='Info.TLabel')
        
        # Image display area
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image Preview", padding="10")
        
        # Original image
        self.original_frame = ttk.Frame(self.image_frame)
        self.original_label = ttk.Label(self.original_frame, text="Original Image", style='Header.TLabel')
        self.original_canvas = tk.Canvas(self.original_frame, width=400, height=300, bg='white')
        
        # Colorized image
        self.colorized_frame = ttk.Frame(self.image_frame)
        self.colorized_label = ttk.Label(self.colorized_frame, text="Colorized Image", style='Header.TLabel')
        self.colorized_canvas = tk.Canvas(self.colorized_frame, width=400, height=300, bg='white')
        
        # Metrics frame
        self.metrics_frame = ttk.LabelFrame(self.main_frame, text="Quality Metrics", padding="10")
        self.metrics_text = tk.Text(self.metrics_frame, height=6, width=50, font=('Courier', 9))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')
    
    def _setup_layout(self):
        """Setup the layout of all widgets."""
        # Main layout
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        self.title_label.pack(pady=(0, 20))
        
        # Control panel
        self.control_frame.pack(fill='x', pady=(0, 20))
        
        # File selection
        self.file_frame.pack(fill='x', pady=(0, 10))
        self.file_label.pack(side='left', padx=(0, 10))
        self.file_button.pack(side='left', padx=(0, 10))
        self.file_path_label.pack(side='left')
        
        # Processing buttons
        self.process_frame.pack(fill='x', pady=(0, 10))
        self.process_button.pack(side='left', padx=(0, 10))
        self.batch_button.pack(side='left')
        
        # Progress
        self.progress_bar.pack(fill='x', pady=(0, 5))
        self.progress_label.pack()
        
        # Image display
        self.image_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Image canvases
        self.original_frame.pack(side='left', padx=(0, 10))
        self.original_label.pack()
        self.original_canvas.pack()
        
        self.colorized_frame.pack(side='right', padx=(10, 0))
        self.colorized_label.pack()
        self.colorized_canvas.pack()
        
        # Metrics
        self.metrics_frame.pack(fill='x')
        self.metrics_text.pack()
        
        # Status bar
        self.status_bar.pack(side='bottom', fill='x')
    
    def _browse_file(self):
        """Open file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.original_path = file_path
            self.file_path_var.set(os.path.basename(file_path))
            self._load_original_image()
            self.process_button.config(state='normal')
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
    
    def _load_original_image(self):
        """Load and display the original image."""
        try:
            # Load image
            image = cv.imread(self.original_path)
            if image is None:
                raise ValueError("Could not load image")
            
            self.original_image = image
            
            # Resize for display
            display_image = self._resize_for_display(image, (400, 300))
            
            # Convert to PIL and display
            display_image_rgb = cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_image_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.original_canvas.delete("all")
            self.original_canvas.create_image(200, 150, image=photo)
            self.original_canvas.image = photo  # Keep reference
            
            # Clear colorized image
            self.colorized_canvas.delete("all")
            self.colorized_image = None
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            logger.error(f"Error loading image: {str(e)}")
    
    def _resize_for_display(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image for display while maintaining aspect ratio."""
        h, w = image.shape[:2]
        target_w, target_h = size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        return cv.resize(image, (new_w, new_h))
    
    def _process_image(self):
        """Process the selected image in a separate thread."""
        if not self.model_loaded or self.original_image is None:
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.process_button.config(state='disabled')
        self.progress_var.set(0)
        self.progress_label.config(text="Processing...")
        self.status_var.set("Processing image...")
        
        # Start processing in separate thread
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Process image in background thread."""
        try:
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(25))
            self.root.after(0, lambda: self.progress_label.config(text="Loading model..."))
            
            # Colorize image
            self.root.after(0, lambda: self.progress_var.set(50))
            self.root.after(0, lambda: self.progress_label.config(text="Colorizing..."))
            
            colorized = self.colorizer.colorize_image(self.original_image)
            self.colorized_image = colorized
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(75))
            self.root.after(0, lambda: self.progress_label.config(text="Displaying result..."))
            
            # Display result
            self.root.after(0, self._display_colorized_image)
            
            # Calculate metrics
            self.root.after(0, lambda: self.progress_var.set(90))
            self.root.after(0, lambda: self.progress_label.config(text="Calculating metrics..."))
            
            metrics = self.colorizer.evaluate_colorization(self.original_image, colorized)
            self.root.after(0, lambda: self._display_metrics(metrics))
            
            # Complete
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.progress_label.config(text="Complete!"))
            self.root.after(0, lambda: self.status_var.set("Processing complete"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            logger.error(f"Processing error: {str(e)}")
        
        finally:
            self.processing = False
            self.root.after(0, lambda: self.process_button.config(state='normal'))
    
    def _display_colorized_image(self):
        """Display the colorized image."""
        if self.colorized_image is None:
            return
        
        try:
            # Resize for display
            display_image = self._resize_for_display(self.colorized_image, (400, 300))
            
            # Convert to PIL and display
            display_image_rgb = cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_image_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.colorized_canvas.delete("all")
            self.colorized_canvas.create_image(200, 150, image=photo)
            self.colorized_canvas.image = photo  # Keep reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display colorized image: {str(e)}")
            logger.error(f"Display error: {str(e)}")
    
    def _display_metrics(self, metrics: dict):
        """Display quality metrics."""
        self.metrics_text.delete(1.0, tk.END)
        
        metrics_text = f"""Quality Metrics:
        
SSIM (Structural Similarity Index): {metrics['ssim']:.4f}
PSNR (Peak Signal-to-Noise Ratio): {metrics['psnr']:.2f} dB
Colorfulness Score: {metrics['colorfulness']:.2f}

Interpretation:
- SSIM: Higher is better (0-1 scale)
- PSNR: Higher is better (typically 20-40 dB is good)
- Colorfulness: Higher indicates more vibrant colors
"""
        
        self.metrics_text.insert(1.0, metrics_text)
    
    def _batch_process(self):
        """Open batch processing dialog."""
        input_dir = filedialog.askdirectory(title="Select Input Directory")
        if not input_dir:
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(input_dir):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(input_dir, file))
        
        if not image_files:
            messagebox.showwarning("Warning", "No image files found in selected directory")
            return
        
        # Confirm batch processing
        result = messagebox.askyesno(
            "Batch Processing",
            f"Process {len(image_files)} images?\nThis may take several minutes."
        )
        
        if result:
            self._start_batch_processing(image_files, output_dir)
    
    def _start_batch_processing(self, image_files: list, output_dir: str):
        """Start batch processing in background thread."""
        thread = threading.Thread(
            target=self._batch_process_thread,
            args=(image_files, output_dir)
        )
        thread.daemon = True
        thread.start()
    
    def _batch_process_thread(self, image_files: list, output_dir: str):
        """Process batch of images in background."""
        try:
            self.root.after(0, lambda: self.status_var.set("Starting batch processing..."))
            
            results = self.colorizer.batch_colorize(image_files, output_dir)
            
            # Show results
            message = f"""Batch processing complete!

Processed: {results['processed']} images
Failed: {results['failed']} images
Total time: {results['total_time']:.2f} seconds
Average time per image: {results['total_time']/results['processed']:.2f} seconds

Output saved to: {output_dir}"""

            self.root.after(0, lambda: messagebox.showinfo("Batch Complete", message))
            self.root.after(0, lambda: self.status_var.set("Batch processing complete"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Batch processing failed: {str(e)}"))
            logger.error(f"Batch processing error: {str(e)}")


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = ModernColorizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 