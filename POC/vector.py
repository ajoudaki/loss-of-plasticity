import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Custom 3D arrow class for better visualization
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# Define the LoP manifold
manifold_normal = np.array([1, 1, 1]) / np.sqrt(3)
manifold_point = np.array([0, 0, 0])

# Create orthonormal basis for the manifold
v1 = np.array([1, 0, -manifold_normal[0]/manifold_normal[2]])
v1 = v1 / np.linalg.norm(v1)
v2 = np.cross(manifold_normal, v1)

def gradient_on_manifold(x, y, z):
    """Gradient that is tangent to the manifold"""
    params = np.array([x, y, z])
    d = np.dot(params - manifold_point, manifold_normal)
    
    # Create gradient with interesting dynamics
    grad = params - manifold_point
    proj = params - d * manifold_normal
    grad[0] += 0.4 * np.cos(2*proj[0]) * np.cos(2*proj[1]) + 0.3*proj[0]
    grad[1] += -0.4 * np.sin(2*proj[0]) * np.sin(2*proj[1]) + 0.3*proj[1]
    
    # Project to tangent space
    grad_tangent = grad - np.dot(grad, manifold_normal) * manifold_normal
    return grad_tangent

# Set up publication-quality parameters
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# Figure 1: Manifold with sparse vector field
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')

# Create manifold mesh
u = np.linspace(-2, 2, 30)
v = np.linspace(-2, 2, 30)
U, V = np.meshgrid(u, v)

X_manifold = manifold_point[0] + U * v1[0] + V * v2[0]
Y_manifold = manifold_point[1] + U * v1[1] + V * v2[1]
Z_manifold = manifold_point[2] + U * v1[2] + V * v2[2]

# Plot manifold
surf = ax1.plot_surface(X_manifold, Y_manifold, Z_manifold, alpha=0.3,
                       color='lightblue', edgecolor='none', linewidth=0)

# Add manifold boundary
boundary_u = np.array([-2, 2, 2, -2, -2])
boundary_v = np.array([-2, -2, 2, 2, -2])
X_boundary = manifold_point[0] + boundary_u[:, None] * v1[0] + boundary_v[:, None] * v2[0]
Y_boundary = manifold_point[1] + boundary_u[:, None] * v1[1] + boundary_v[:, None] * v2[1]
Z_boundary = manifold_point[2] + boundary_u[:, None] * v1[2] + boundary_v[:, None] * v2[2]

for i in range(len(boundary_u)-1):
    ax1.plot([X_boundary[i], X_boundary[i+1]], 
             [Y_boundary[i], Y_boundary[i+1]], 
             [Z_boundary[i], Z_boundary[i+1]], 
             'b-', linewidth=2, alpha=0.6)

# Sparse vector field (5x5 grid)
n_arrows = 5
u_sample = np.linspace(-1.2, 1.2, n_arrows)
v_sample = np.linspace(-1.2, 1.2, n_arrows)

for i, u_val in enumerate(u_sample):
    for j, v_val in enumerate(v_sample):
        # Point on manifold
        point = manifold_point + u_val * v1 + v_val * v2
        
        # Gradient at this point
        grad = gradient_on_manifold(point[0], point[1], point[2])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm * 0.4  # Normalize length
        
        # Use custom 3D arrow for better visibility
        arrow = Arrow3D([point[0], point[0] - grad[0]],
                       [point[1], point[1] - grad[1]],
                       [point[2], point[2] - grad[2]],
                       mutation_scale=20, lw=2, arrowstyle='->', color='red', alpha=0.8)
        ax1.add_artist(arrow)

# Add normal vector
normal_base = manifold_point + 0.5 * v1 + 0.5 * v2
normal_arrow = Arrow3D([normal_base[0], normal_base[0] + 0.8 * manifold_normal[0]],
                      [normal_base[1], normal_base[1] + 0.8 * manifold_normal[1]],
                      [normal_base[2], normal_base[2] + 0.8 * manifold_normal[2]],
                      mutation_scale=20, lw=3, arrowstyle='->', color='green', alpha=0.8)
ax1.add_artist(normal_arrow)

# Add a sample tangent vector for comparison
tangent = -gradient_on_manifold(normal_base[0], normal_base[1], normal_base[2])
tangent = tangent / np.linalg.norm(tangent) * 0.8
tangent_arrow = Arrow3D([normal_base[0], normal_base[0] + tangent[0]],
                       [normal_base[1], normal_base[1] + tangent[1]],
                       [normal_base[2], normal_base[2] + tangent[2]],
                       mutation_scale=20, lw=3, arrowstyle='->', color='blue', alpha=0.8)
ax1.add_artist(tangent_arrow)

# Labels
ax1.text(normal_base[0] + 0.9 * manifold_normal[0], 
         normal_base[1] + 0.9 * manifold_normal[1],
         normal_base[2] + 0.9 * manifold_normal[2], 
         r'$\mathbf{n}$', fontsize=16, color='green')

ax1.text(normal_base[0] + tangent[0] + 0.1, 
         normal_base[1] + tangent[1] + 0.1,
         normal_base[2] + tangent[2] + 0.1, 
         r'$\nabla_\theta\mathcal{L}$', fontsize=16, color='blue')

ax1.set_xlabel(r'$\theta_1$')
ax1.set_ylabel(r'$\theta_2$')
ax1.set_zlabel(r'$\theta_3$')
ax1.set_title('LoP Manifold $\mathcal{M}$ with Tangent Gradient Field', pad=20)

# Set viewing angle
ax1.view_init(elev=25, azim=45)
ax1.set_box_aspect([1,1,0.8])

# Remove grid for cleaner look
ax1.grid(False)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig('lop_manifold_vector_field.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Figure 2: Gradient trajectories on manifold
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')

# Plot manifold
ax2.plot_surface(X_manifold, Y_manifold, Z_manifold, alpha=0.2,
                color='lightblue', edgecolor='none')

# Add boundary
for i in range(len(boundary_u)-1):
    ax2.plot([X_boundary[i], X_boundary[i+1]], 
             [Y_boundary[i], Y_boundary[i+1]], 
             [Z_boundary[i], Z_boundary[i+1]], 
             'b-', linewidth=2, alpha=0.6)

# Gradient descent trajectories
n_trajectories = 4
np.random.seed(42)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Starting points in a pattern
start_points = [
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0)
]

for traj_idx in range(n_trajectories):
    u_start, v_start = start_points[traj_idx]
    start_point = manifold_point + u_start * v1 + v_start * v2
    
    # Gradient descent
    trajectory = [start_point]
    current = start_point.copy()
    learning_rate = 0.08
    n_steps = 50
    
    for step in range(n_steps):
        grad = gradient_on_manifold(current[0], current[1], current[2])
        current = current - learning_rate * grad
        trajectory.append(current.copy())
    
    trajectory = np.array(trajectory)
    
    # Plot trajectory
    ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
             color=colors[traj_idx], linewidth=2.5, alpha=0.9)
    
    # Start point
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                color=colors[traj_idx], s=80, marker='o', edgecolor='black',
                linewidth=2)
    
    # End point
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                color=colors[traj_idx], s=100, marker='*', edgecolor='black',
                linewidth=2)

# Add a few gradient vectors
sparse_n = 3
u_sparse = np.linspace(-0.8, 0.8, sparse_n)
v_sparse = np.linspace(-0.8, 0.8, sparse_n)

for i, u_val in enumerate(u_sparse):
    for j, v_val in enumerate(v_sparse):
        point = manifold_point + u_val * v1 + v_val * v2
        grad = gradient_on_manifold(point[0], point[1], point[2])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad = grad / grad_norm * 0.3
        
        arrow = Arrow3D([point[0], point[0] - grad[0]],
                       [point[1], point[1] - grad[1]],
                       [point[2], point[2] - grad[2]],
                       mutation_scale=15, lw=1.5, arrowstyle='->', 
                       color='red', alpha=0.5)
        ax2.add_artist(arrow)

ax2.set_xlabel(r'$\theta_1$')
ax2.set_ylabel(r'$\theta_2$')
ax2.set_zlabel(r'$\theta_3$')
ax2.set_title('Gradient Descent Trajectories Confined to $\mathcal{M}$', pad=20)

ax2.view_init(elev=30, azim=-60)
ax2.set_box_aspect([1,1,0.8])

ax2.grid(False)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig('lop_manifold_trajectories.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Figure 3: Conceptual diagram - side view
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')

# Smaller manifold patch for clarity
u_small = np.linspace(-1, 1, 20)
v_small = np.linspace(-1, 1, 20)
U_small, V_small = np.meshgrid(u_small, v_small)

X_small = manifold_point[0] + U_small * v1[0] + V_small * v2[0]
Y_small = manifold_point[1] + U_small * v1[1] + V_small * v2[1]
Z_small = manifold_point[2] + U_small * v1[2] + V_small * v2[2]

ax3.plot_surface(X_small, Y_small, Z_small, alpha=0.4,
                color='lightblue', edgecolor='none')

# Key point demonstration
demo_point = manifold_point

# Normal vector
ax3.quiver(demo_point[0], demo_point[1], demo_point[2],
          manifold_normal[0], manifold_normal[1], manifold_normal[2],
          color='green', alpha=1, arrow_length_ratio=0.15, linewidth=4)

# Gradient vector (tangent)
grad_demo = gradient_on_manifold(demo_point[0], demo_point[1], demo_point[2])
grad_demo = grad_demo / np.linalg.norm(grad_demo) * 1.0
ax3.quiver(demo_point[0], demo_point[1], demo_point[2],
          -grad_demo[0], -grad_demo[1], -grad_demo[2],
          color='red', alpha=1, arrow_length_ratio=0.15, linewidth=4)

# Another tangent vector
tangent2 = np.cross(grad_demo, manifold_normal)
tangent2 = tangent2 / np.linalg.norm(tangent2) * 1.0
ax3.quiver(demo_point[0], demo_point[1], demo_point[2],
          tangent2[0], tangent2[1], tangent2[2],
          color='blue', alpha=1, arrow_length_ratio=0.15, linewidth=4)

# Labels with better positioning
ax3.text(demo_point[0] + 1.2 * manifold_normal[0], 
         demo_point[1] + 1.2 * manifold_normal[1],
         demo_point[2] + 1.2 * manifold_normal[2], 
         r'$\mathbf{n} \perp \mathcal{M}$', fontsize=14, color='green', ha='center')

ax3.text(demo_point[0] - 1.2 * grad_demo[0], 
         demo_point[1] - 1.2 * grad_demo[1],
         demo_point[2] - 1.2 * grad_demo[2], 
         r'$-\nabla_\theta\mathcal{L} \in T_\theta\mathcal{M}$', fontsize=14, color='red', ha='center')

ax3.text(demo_point[0] + 1.2 * tangent2[0], 
         demo_point[1] + 1.2 * tangent2[1],
         demo_point[2] + 1.2 * tangent2[2], 
         r'$T_\theta\mathcal{M}$', fontsize=14, color='blue', ha='center')

# Add coordinate frame origin
ax3.scatter(demo_point[0], demo_point[1], demo_point[2], 
           color='black', s=100, marker='o')

ax3.set_xlabel(r'$\theta_1$')
ax3.set_ylabel(r'$\theta_2$')
ax3.set_zlabel(r'$\theta_3$')
ax3.set_title(r'Key Property: $\nabla_\theta\mathcal{L}(\theta) \in T_\theta\mathcal{M}$ for all $\theta \in \mathcal{M}$', 
              pad=20)

ax3.view_init(elev=15, azim=30)
ax3.set_box_aspect([1,1,0.8])

# Minimal grid
ax3.grid(True, alpha=0.2)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False

plt.tight_layout()
plt.savefig('lop_manifold_concept.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Three PDF figures created:")
print("1. lop_manifold_vector_field.pdf - Shows the manifold with sparse gradient field")
print("2. lop_manifold_trajectories.pdf - Shows confined gradient descent paths")
print("3. lop_manifold_concept.pdf - Illustrates the key tangency property")
print("\nFigures are designed for publication with:")
print("- Clean, uncluttered 3D visualization")
print("- Sparse but informative vector fields")
print("- Clear mathematical notation")
print("- High-quality PDF output at 300 DPI")