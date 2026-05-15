import cv2
import numpy as np
from scipy import ndimage


class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self._undo_stack = []          # list of numpy arrays (previous states)
        self._max_undo = 20            # keep last 20 steps

    def _push_undo(self):
        """Save current state before applying an operation."""
        if self.current_image is not None:
            self._undo_stack.append(self.current_image.copy())
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)

    def undo(self):
        """Revert to the previous state. Returns (image, success)."""
        if not self._undo_stack:
            return self.current_image, False
        self.current_image = self._undo_stack.pop()
        return self.current_image, True

    def can_undo(self):
        return len(self._undo_stack) > 0

    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        self.current_image = self.original_image.copy()
        self._undo_stack.clear()
        return self.current_image

    def load_from_array(self, img_array):
        self.original_image = img_array.copy()
        self.current_image = img_array.copy()
        self._undo_stack.clear()
        return self.current_image

    def reset(self):
        if self.original_image is not None:
            self._push_undo()
            self.current_image = self.original_image.copy()
        return self.current_image

    def get_current(self):
        return self.current_image

    # ─────────────────────────────────────────────
    # 1. POINT OPERATIONS
    # ─────────────────────────────────────────────

    def addition(self, value=50):
        """Add a scalar value to every pixel."""
        self._push_undo()
        img = self.current_image.astype(np.int32)
        img = np.clip(img + value, 0, 255).astype(np.uint8)
        self.current_image = img
        return self.current_image

    def subtraction(self, value=50):
        """Subtract a scalar value from every pixel."""
        self._push_undo()
        img = self.current_image.astype(np.int32)
        img = np.clip(img - value, 0, 255).astype(np.uint8)
        self.current_image = img
        return self.current_image

    def division(self, value=2):
        """Divide every pixel by a scalar value (value mapped from 1-255 → 1-10)."""
        self._push_undo()
        divisor = max(1, value / 25.5)
        img = self.current_image.astype(np.float32)
        img = np.clip(img / divisor, 0, 255).astype(np.uint8)
        self.current_image = img
        return self.current_image

    def complement(self):
        """Compute the complement (negative) of the image."""
        self._push_undo()
        self.current_image = 255 - self.current_image
        return self.current_image

    # ─────────────────────────────────────────────
    # 2. COLOR IMAGE OPERATIONS
    # ─────────────────────────────────────────────

    def change_red_lighting(self, value=50):
        self._push_undo()
        img = self.current_image.copy().astype(np.int32)
        img[:, :, 2] = np.clip(img[:, :, 2] + value, 0, 255)
        self.current_image = img.astype(np.uint8)
        return self.current_image

    def swap_r_to_g(self):
        self._push_undo()
        img = self.current_image.copy()
        img[:, :, 1], img[:, :, 2] = (
            self.current_image[:, :, 2].copy(),
            self.current_image[:, :, 1].copy(),
        )
        self.current_image = img
        return self.current_image

    def eliminate_red(self):
        self._push_undo()
        img = self.current_image.copy()
        img[:, :, 2] = 0
        self.current_image = img
        return self.current_image

    # ─────────────────────────────────────────────
    # 3. IMAGE HISTOGRAM
    # ─────────────────────────────────────────────

    def histogram_stretching_gray(self):
        self._push_undo()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        min_val = gray.min()
        max_val = gray.max()
        if max_val == min_val:
            stretched = gray
        else:
            stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        self.current_image = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        return self.current_image

    def histogram_equalization_gray(self):
        self._push_undo()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        self.current_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        return self.current_image

    # ─────────────────────────────────────────────
    # 4. NEIGHBORHOOD PROCESSING
    # ─────────────────────────────────────────────

    def _kernel_size(self, value):
        """Convert slider value (0-255) to an odd kernel size (1-21)."""
        k = max(1, int(value / 255 * 10)) * 2 + 1
        return k

    def average_filter(self, value=50):
        self._push_undo()
        k = self._kernel_size(value)
        self.current_image = cv2.blur(self.current_image, (k, k))
        return self.current_image

    def laplacian_filter(self, value=50):
        self._push_undo()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        self.current_image = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
        return self.current_image

    def maximum_filter(self, value=50):
        self._push_undo()
        k = self._kernel_size(value)
        kernel = np.ones((k, k), np.uint8)
        self.current_image = cv2.dilate(self.current_image, kernel)
        return self.current_image

    def minimum_filter(self, value=50):
        self._push_undo()
        k = self._kernel_size(value)
        kernel = np.ones((k, k), np.uint8)
        self.current_image = cv2.erode(self.current_image, kernel)
        return self.current_image

    def median_filter(self, value=50):
        self._push_undo()
        k = self._kernel_size(value)
        if k % 2 == 0:
            k += 1
        self.current_image = cv2.medianBlur(self.current_image, k)
        return self.current_image

    def mode_filter(self, value=50):
        self._push_undo()
        k = self._kernel_size(value)
        result = np.zeros_like(self.current_image)
        for c in range(self.current_image.shape[2]):
            result[:, :, c] = ndimage.generic_filter(
                self.current_image[:, :, c],
                lambda x: np.bincount(x.astype(np.int32)).argmax(),
                size=k,
            )
        self.current_image = result
        return self.current_image

    # ─────────────────────────────────────────────
    # 5. IMAGE RESTORATION
    # ─────────────────────────────────────────────

    def add_salt_pepper_noise(self, value=50):
        self._push_undo()
        img = self.current_image.copy()
        amount = value / 255 * 0.1
        num_salt = int(amount * img.size * 0.5)
        num_pepper = int(amount * img.size * 0.5)
        coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
        img[coords[0], coords[1]] = 255
        coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
        img[coords[0], coords[1]] = 0
        self.current_image = img
        return self.current_image

    def salt_pepper_average_filter(self, value=50):
        self._push_undo()
        k = self._kernel_size(value)
        self.current_image = cv2.blur(self.current_image, (k, k))
        return self.current_image

    def salt_pepper_median_filter(self, value=50):
        self._push_undo()
        k = self._kernel_size(value)
        if k % 2 == 0:
            k += 1
        self.current_image = cv2.medianBlur(self.current_image, k)
        return self.current_image

    def outlier_method(self, value=50):
        self._push_undo()
        threshold = value
        img = self.current_image.astype(np.float32)
        blurred = cv2.blur(img, (3, 3))
        diff = np.abs(img - blurred)
        mask = diff > threshold
        result = np.where(mask, blurred, img)
        self.current_image = np.clip(result, 0, 255).astype(np.uint8)
        return self.current_image

    def add_gaussian_noise(self, value=50):
        self._push_undo()
        sigma = value / 255 * 50
        noise = np.random.normal(0, sigma, self.current_image.shape)
        img = np.clip(self.current_image.astype(np.float32) + noise, 0, 255)
        self.current_image = img.astype(np.uint8)
        return self.current_image

    def gaussian_image_averaging(self, value=50):
        self._push_undo()
        n = max(2, int(value / 25))
        acc = self.current_image.astype(np.float32)
        for _ in range(n - 1):
            noise = np.random.normal(0, 20, self.current_image.shape)
            noisy = np.clip(self.current_image.astype(np.float32) + noise, 0, 255)
            acc += noisy
        self.current_image = (acc / n).astype(np.uint8)
        return self.current_image

    def gaussian_average_filter(self, value=50):
        self._push_undo()
        k = self._kernel_size(value)
        sigma = value / 50
        self.current_image = cv2.GaussianBlur(self.current_image, (k, k), sigma)
        return self.current_image

    # ─────────────────────────────────────────────
    # 6. IMAGE SEGMENTATION
    # ─────────────────────────────────────────────

    def basic_global_thresholding(self, value=127):
        self._push_undo()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
        self.current_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return self.current_image

    def automatic_thresholding(self, value=None):
        self._push_undo()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.current_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return self.current_image

    def adaptive_thresholding(self, value=50):
        self._push_undo()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        block = self._kernel_size(value)
        if block < 3:
            block = 3
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 2
        )
        self.current_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return self.current_image

    def sobel_detector(self, value=50):
        self._push_undo()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        ksize = self._kernel_size(value)
        if ksize > 7:
            ksize = 7
        if ksize % 2 == 0:
            ksize += 1
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        self.current_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        return self.current_image

    def dilation(self, value=50):
        self._push_undo()
        se = self._struct_element(value)
        self.current_image = cv2.dilate(self.current_image, se)
        return self.current_image

    def erosion(self, value=50):
        self._push_undo()
        se = self._struct_element(value)
        self.current_image = cv2.erode(self.current_image, se)
        return self.current_image

    def opening(self, value=50):
        self._push_undo()
        se = self._struct_element(value)
        self.current_image = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, se)
        return self.current_image

    def closing(self, value=50):
        self._push_undo()
        se = self._struct_element(value)
        self.current_image = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, se)
        return self.current_image

    def internal_boundary(self, value=50):
        self._push_undo()
        se = self._struct_element(value)
        eroded = cv2.erode(self.current_image, se)
        self.current_image = cv2.subtract(self.current_image, eroded)
        return self.current_image

    def external_boundary(self, value=50):
        self._push_undo()
        se = self._struct_element(value)
        dilated = cv2.dilate(self.current_image, se)
        self.current_image = cv2.subtract(dilated, self.current_image)
        return self.current_image

    def morphological_gradient(self, value=50):
        self._push_undo()
        se = self._struct_element(value)
        self.current_image = cv2.morphologyEx(
            self.current_image, cv2.MORPH_GRADIENT, se
        )
        return self.current_image

    def morphological_gradient(self, value=50):
        """Morphological gradient = dilated - eroded."""
        se = self._struct_element(value)
        self.current_image = cv2.morphologyEx(
            self.current_image, cv2.MORPH_GRADIENT, se
        )
        return self.current_image
