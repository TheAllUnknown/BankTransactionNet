import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha1 = 2
alpha2 = 3
x_min = 1  # Minimum value of x
x_max = 100  # Maximum value of x
num_points = 1000  # Number of points to plot

# Generate x values
x = np.linspace(x_min, x_max, num_points)

# Compute the power law values for both alpha
y1 = x**(-alpha1)
y2 = x**(-alpha2)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot for alpha = 2
plt.plot(x, y1, label=f'Power Law: $x^{{-{alpha1}}}$', color='blue')

# Plot for alpha = 3
plt.plot(x, y2, label=f'Power Law: $x^{{-{alpha2}}}$', color='red')

# Add labels and title
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Power Law Distributions with alpha = 2 and alpha = 3')
plt.yscale('log')  # Use logarithmic scale for better visualization
plt.xscale('log')  # Use logarithmic scale for better visualization
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.show()
