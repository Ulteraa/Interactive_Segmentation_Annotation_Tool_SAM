import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Adjust subplot to make room for the button
line, = ax.plot(x, y)

# Define the button's location and size [left, bottom, width, height]
button_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
button = Button(button_ax, 'Update')

# Define the action to be taken when the button is clicked
def update(event):
    y_new = np.cos(x)  # New data to plot
    line.set_ydata(y_new)
    plt.draw()

# Link the button with the update function
button.on_clicked(update)

plt.show()
