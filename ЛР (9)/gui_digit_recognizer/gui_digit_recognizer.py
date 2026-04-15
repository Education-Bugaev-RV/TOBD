import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

# Путь к модели относительно расположения скрипта
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.keras")
model = keras.models.load_model(model_path)
root = tk.Tk()
root.title("Распознавание цифры")

canvas_width = 280
canvas_height = 280

brush_size = 10

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="black")
canvas.pack(pady=10)

image = Image.new("L", (canvas_width, canvas_height), color=0)
draw = ImageDraw.Draw(image)

def draw_digit(event):
    x = event.x
    y = event.y

    canvas.create_oval(
        x - brush_size,
        y - brush_size,
        x + brush_size,
        y + brush_size,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (
            x - brush_size,
            y - brush_size,
            x + brush_size,
            y + brush_size
        ),
        fill=255
    )

canvas.bind("<B1-Motion>", draw_digit)

def clear_canvas():
    global image, draw

    canvas.delete("all")
    canvas.configure(bg="black")

    image = Image.new("L", (canvas_width, canvas_height), color=0)
    draw = ImageDraw.Draw(image)
    
    result_label.config(text="Здесь будет результат")


def recognize_digit():
    img_array = np.array(image)
    coords = np.argwhere(img_array > 20)

    if len(coords) == 0:
        result_label.config(text="Сначала нарисуйте цифру")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = img_array[y_min:y_max + 1, x_min:x_max + 1]

    digit_image = Image.fromarray(digit)

    w, h = digit_image.size

    max_side = max(w, h)
    square_image = Image.new("L", (max_side, max_side), color=0)

    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    square_image.paste(digit_image, (paste_x, paste_y))

    final_size = max_side + 20

    padded_image = Image.new("L", (final_size, final_size), color=0)
    padded_image.paste(square_image, (10, 10))

    img_resized = padded_image.resize((28, 28))
    img_resized.save("drawn.png")
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction)

    result_label.config(
        text=f"Предсказание: {predicted_digit} | Уверенность: {confidence:.2f}"
    )

buttons_frame = tk.Frame(root)
buttons_frame.pack(pady=10)

clear_button = tk.Button(buttons_frame, text="Очистить", command=clear_canvas, width=15)
clear_button.pack(side="left", padx=10)

recognize_button = tk.Button(buttons_frame, text="Распознать", command=recognize_digit, width=15)
recognize_button.pack(side="left", padx=10)

result_label = tk.Label(root, text="Здесь будет результат", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
