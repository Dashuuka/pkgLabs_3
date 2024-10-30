import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import sys

# Получаем текущую директорию, где находится EXE
base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Путь к папке с тестовыми изображениями
test_images_path = os.path.join(base_path, 'Тесты')

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Приложение для обработки изображений")

        # Настройка Canvas с прокруткой
        self.canvas = tk.Canvas(root)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas)

        # Размещаем Canvas и прокрутку
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Фреймы для размещения элементов (внутри прокручиваемого фрейма)
        self.left_frame = tk.Frame(self.scrollable_frame)
        self.right_frame = tk.Frame(self.scrollable_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Виджеты левой стороны
        self.load_left_image_button = tk.Button(self.left_frame, text="Загрузить изображение", command=self.load_left_image_method)
        self.load_left_image_button.pack()

        self.left_method_label = tk.Label(self.left_frame, text="Выберите метод")
        self.left_method_label.pack()

        self.left_method_var = tk.StringVar()
        self.left_method_var.set("Нелинейный фильтр (статистика порядка)")
        self.left_method_menu = tk.OptionMenu(self.left_frame, self.left_method_var,
                                              "Нелинейный фильтр (статистика порядка)", "Медианный фильтр",
                                              "Максимальный фильтр", "Минимальный фильтр")
        self.left_method_menu.pack()

        self.left_process_button = tk.Button(self.left_frame, text="Обработать", command=self.process_left_image_method)
        self.left_process_button.pack()

        self.left_original_image_label = tk.Label(self.left_frame, text="Исходное изображение:")
        self.left_original_image_label.pack()
        self.left_image_label = tk.Label(self.left_frame)
        self.left_image_label.pack()

        self.left_processed_image_label = tk.Label(self.left_frame, text="Обработанное изображение:")
        self.left_processed_image_label.pack()
        self.left_processed_image_display = tk.Label(self.left_frame)
        self.left_processed_image_display.pack()

        # Добавляем кнопку для сохранения обработанного изображения слева
        self.save_left_image_button = tk.Button(self.left_frame, text="Сохранить обработанное изображение", command=self.save_left_image)
        self.save_left_image_button.pack()

        # Виджеты правой стороны
        self.load_right_image_button = tk.Button(self.right_frame, text="Загрузить изображение",
                                                 command=self.load_right_image_method)
        self.load_right_image_button.pack()

        self.right_method_label = tk.Label(self.right_frame, text="Выберите метод")
        self.right_method_label.pack()

        self.structuring_element_label = tk.Label(self.right_frame, text="Выберите структурирующий элемент")
        self.structuring_element_label.pack()

        self.structuring_element_var = tk.StringVar()
        self.structuring_element_var.set("Прямоугольник")
        self.structuring_element_menu = tk.OptionMenu(self.right_frame, self.structuring_element_var, "Прямоугольник",
                                                      "Эллипс", "Крест")
        self.structuring_element_menu.pack()

        self.kernel_size_label = tk.Label(self.right_frame, text="Размер ядра (например, 5)")
        self.kernel_size_label.pack()

        self.kernel_size_entry = tk.Entry(self.right_frame)
        self.kernel_size_entry.pack()

        self.right_method_var = tk.StringVar()
        self.right_method_var.set("Морфологическая операция")
        self.right_method_menu = tk.OptionMenu(self.right_frame, self.right_method_var,
                                               "Дилатация", "Эрозия", "Открытие", "Закрытие")
        self.right_method_menu.pack()

        self.right_process_button = tk.Button(self.right_frame, text="Обработать", command=self.process_right_image_method)
        self.right_process_button.pack()

        self.right_original_image_label = tk.Label(self.right_frame, text="Исходное изображение:")
        self.right_original_image_label.pack()
        self.right_image_label = tk.Label(self.right_frame)
        self.right_image_label.pack()

        self.right_processed_image_label = tk.Label(self.right_frame, text="Обработанное изображение:")
        self.right_processed_image_label.pack()
        self.right_processed_image_display = tk.Label(self.right_frame)
        self.right_processed_image_display.pack()

        # Добавляем кнопку для сохранения обработанного изображения справа
        self.save_right_image_button = tk.Button(self.right_frame, text="Сохранить обработанное изображение", command=self.save_right_image)
        self.save_right_image_button.pack()

        # Изображения
        self.left_image = None
        self.right_image = None
        self.left_processed_image = None
        self.right_processed_image = None

    def load_left_image_method(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.left_image = Image.open(file_path)
            self.display_image(self.left_image, self.left_image_label)

    def load_right_image_method(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.right_image = Image.open(file_path)
            self.display_image(self.right_image, self.right_image_label)

    def display_image(self, image, label):
        screen_width, screen_height = self.root.winfo_screenwidth() // 2, self.root.winfo_screenheight() // 2
        image.thumbnail((screen_width, screen_height))
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def process_left_image_method(self):
        if self.left_image is None:
            messagebox.showerror("Ошибка", "Изображение не загружено!")
            return
        method = self.left_method_var.get()
        if method == "Нелинейный фильтр (статистика порядка)":
            self.left_processed_image = self.apply_order_statistics_filter(self.left_image)
        elif method == "Медианный фильтр":
            self.left_processed_image = self.apply_median_filter(self.left_image)
        elif method == "Максимальный фильтр":
            self.left_processed_image = self.apply_max_filter(self.left_image)
        elif method == "Минимальный фильтр":
            self.left_processed_image = self.apply_min_filter(self.left_image)
        else:
            messagebox.showerror("Ошибка", "Неверный метод выбран!")
            return
        self.display_image(self.left_processed_image, self.left_processed_image_display)

    def process_right_image_method(self):
        if self.right_image is None:
            messagebox.showerror("Ошибка", "Изображение не загружено!")
            return
        method = self.right_method_var.get()
        if method == "Дилатация":
            self.right_processed_image = self.apply_morphological_operation(self.right_image, cv2.MORPH_DILATE)
        elif method == "Эрозия":
            self.right_processed_image = self.apply_morphological_operation(self.right_image, cv2.MORPH_ERODE)
        elif method == "Открытие":
            self.right_processed_image = self.apply_morphological_operation(self.right_image, cv2.MORPH_OPEN)
        elif method == "Закрытие":
            self.right_processed_image = self.apply_morphological_operation(self.right_image, cv2.MORPH_CLOSE)
        else:
            messagebox.showerror("Ошибка", "Неверный метод выбран!")
            return
        self.display_image(self.right_processed_image, self.right_processed_image_display)

    def apply_order_statistics_filter(self, image):
        img = np.array(image)
        filtered_img = cv2.medianBlur(img, 5)
        return Image.fromarray(filtered_img)

    def apply_median_filter(self, image):
        img = np.array(image)
        filtered_img = cv2.medianBlur(img, 5)
        return Image.fromarray(filtered_img)

    def apply_max_filter(self, image):
        img = np.array(image)
        kernel = np.ones((5, 5), np.uint8)
        filtered_img = cv2.dilate(img, kernel)
        return Image.fromarray(filtered_img)

    def apply_min_filter(self, image):
        img = np.array(image)
        filtered_img = cv2.morphologyEx(img, cv2.MORPH_MIN, np.ones((5, 5), np.uint8))
        return Image.fromarray(filtered_img)

    def apply_morphological_operation(self, image, operation):
        try:
            kernel_size = int(self.kernel_size_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Неверный размер ядра! Введите положительное целое число.")
            return image

        if kernel_size <= 0:
            messagebox.showerror("Ошибка", "Размер ядра должен быть положительным числом!")
            return image

        element_type = self.structuring_element_var.get()
        if element_type == "Прямоугольник":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif element_type == "Эллипс":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif element_type == "Крест":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        else:
            messagebox.showerror("Ошибка", "Неверный тип структурирующего элемента!")
            return image

        img = np.array(image)
        morphed_img = cv2.morphologyEx(img, operation, kernel)
        return Image.fromarray(morphed_img)

    def save_left_image(self):
        if self.left_processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All Files", "*.*")])
            if file_path:
                self.left_processed_image.save(file_path)
                messagebox.showinfo("Сохранение", "Изображение успешно сохранено!")

    def save_right_image(self):
        if self.right_processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All Files", "*.*")])
            if file_path:
                self.right_processed_image.save(file_path)
                messagebox.showinfo("Сохранение", "Изображение успешно сохранено!")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
