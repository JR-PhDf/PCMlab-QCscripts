import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import re

# Element color mapping
element_colors = {
    'H': 'white',
    'C': 'gray',
    'N': 'blue',
    'O': 'red',
    'P': 'orange',
    'S': 'yellow',
    'F': 'lightgreen',
    'Cl': 'green',
    'Br': 'brown'
}

# Atomic radius of an element (used for the size of a sphere)
element_radii = {
    'H': 30,
    'C': 70,
    'N': 65,
    'O': 60,
    'P': 100,
    'S': 100,
    'F': 50,
    'Cl': 100,
    'Br': 115
}

# Covalent radius (used to determine bonding)
covalent_radii = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'P': 1.07,
    'S': 1.05,
    'F': 0.57,
    'Cl': 1.02,
    'Br': 1.20
}


def parse_xyz_dipole_file(filename):
    """Parse files containing xyz coordinates and dipole moments"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Deciphering molecular names
    molecule_name = lines[0].strip()
    if molecule_name.startswith('#'):
        molecule_name = molecule_name[1:].strip()

    # Analysis of the number of atoms
    atom_count = int(lines[1].strip())

    # Analyzing atomic coordinates
    atoms = []
    for i in range(3, 3 + atom_count):
        parts = lines[i].strip().split()
        if len(parts) >= 4:
            element = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            atoms.append({'element': element, 'x': x, 'y': y, 'z': z})

    # Analytical dipole moment
    dipole = {}
    for line in lines[3 + atom_count:]:
        if 'X=' in line and 'Y=' in line and 'Z=' in line:
            # Extracting values ​​using regular expressions
            x_match = re.search(r'X=\s*([-\d.]+)', line)
            y_match = re.search(r'Y=\s*([-\d.]+)', line)
            z_match = re.search(r'Z=\s*([-\d.]+)', line)
            tot_match = re.search(r'Tot=\s*([-\d.]+)', line)

            if x_match and y_match and z_match and tot_match:
                dipole = {
                    'x': float(x_match.group(1)),
                    'y': float(y_match.group(1)),
                    'z': float(z_match.group(1)),
                    'total': float(tot_match.group(1))
                }
                break

    return molecule_name, atoms, dipole


def calculate_bonds(atoms, max_bond_factor=1.3):
    """Calculate the chemical bonds between atoms"""
    bonds = []
    n_atoms = len(atoms)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            atom1 = atoms[i]
            atom2 = atoms[j]

            # Calculate distance
            distance = np.sqrt(
                (atom2['x'] - atom1['x']) ** 2 +
                (atom2['y'] - atom1['y']) ** 2 +
                (atom2['z'] - atom1['z']) ** 2
            )

            # Determine if bonding occurs
            max_bond_length = (
                                      covalent_radii.get(atom1['element'], 0.7) +
                                      covalent_radii.get(atom2['element'], 0.7)
                              ) * max_bond_factor

            if distance < max_bond_length:
                bonds.append((i, j))

    return bonds


def visualize_molecule_with_dipole(molecule_name, atoms, dipole, bonds,
                                   axis_label_size=14, tick_label_size=12,
                                   title_size=16, element_label_size=10,
                                   fig_size=(12, 10)):
    """
    Visualization of molecular structure and dipole moment

    parameter:
    - axis_label_size: Axis label font size
    - tick_label_size: Axis tick label font size
    - title_size: Title font size
    - element_label_size: Element tag font size
    - fig_size:Graphic dimensions (width, height)

    Notice：
    - Atomic coordinate units：Å (angstrom)
    - Dipole moment unit：Debye
    - The dipole moment vector is used only to indicate direction; its length is scaled for easier visualization.
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # Extracting atomic coordinates
    x_coords = [atom['x'] for atom in atoms]
    y_coords = [atom['y'] for atom in atoms]
    z_coords = [atom['z'] for atom in atoms]

    # Calculation of molecular center
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    center_z = np.mean(z_coords)

    # Centralized coordinates
    x_coords = [x - center_x for x in x_coords]
    y_coords = [y - center_y for y in y_coords]
    z_coords = [z - center_z for z in z_coords]

    # Drawing chemical bonds
    for bond in bonds:
        i, j = bond
        ax.plot([x_coords[i], x_coords[j]],
                [y_coords[i], y_coords[j]],
                [z_coords[i], z_coords[j]],
                'k-', linewidth=2, alpha=0.6)

    # Drawing atoms
    for i, atom in enumerate(atoms):
        color = element_colors.get(atom['element'], 'gray')
        size = element_radii.get(atom['element'], 50)
        ax.scatter(x_coords[i], y_coords[i], z_coords[i],
                   c=color, s=size, edgecolors='black', linewidths=1, alpha=0.8)

        # Add element tags (font size can be adjusted)
        ax.text(x_coords[i], y_coords[i], z_coords[i] + 0.3,
                atom['element'], fontsize=element_label_size, ha='center')

    # Draw the dipole moment vector
    origin = [0, 0, 0]
    scale = 2  # Vector scaling factor (for visualization purposes only, does not represent actual unit conversion)

    # Note: The unit of dipole moment is Debye, while the unit of the coordinate axes is Å.
    # The scale here is only used to make vectors clearly visible in the graph, not for unit conversion.

    # X component (red)
    if abs(dipole['x']) > 0.001:
        ax.quiver(origin[0], origin[1], origin[2],
                  dipole['x'] * scale, 0, 0,
                  color='red', arrow_length_ratio=0.3, linewidth=3,
                  label=f'X: {dipole["x"]:.3f} D')

    # Y component (green)
    if abs(dipole['y']) > 0.001:
        ax.quiver(origin[0], origin[1], origin[2],
                  0, dipole['y'] * scale, 0,
                  color='green', arrow_length_ratio=0.3, linewidth=3,
                  label=f'Y: {dipole["y"]:.3f} D')

    # Z component (blue)
    if abs(dipole['z']) > 0.001:
        ax.quiver(origin[0], origin[1], origin[2],
                  0, 0, dipole['z'] * scale,
                  color='blue', arrow_length_ratio=0.3, linewidth=3,
                  label=f'Z: {dipole["z"]:.3f} D')

    # Total dipole moment (yellow)
    ax.quiver(origin[0], origin[1], origin[2],
              dipole['x'] * scale, dipole['y'] * scale, dipole['z'] * scale,
              color='gold', arrow_length_ratio=0.3, linewidth=4,
              label=f'Total: {dipole["total"]:.3f} D')

    # Set graphic properties - Adjust font size
    ax.set_xlabel('X (Å)', fontsize=axis_label_size, labelpad=10)
    ax.set_ylabel('Y (Å)', fontsize=axis_label_size, labelpad=10)
    ax.set_zlabel('Z (Å)', fontsize=axis_label_size, labelpad=10)
    ax.set_title(f'{molecule_name}\nDipole Moment Visualization', fontsize=title_size, pad=20)

    # Set the font size of the axis tick labels
    ax.tick_params(axis='x', labelsize=tick_label_size)
    ax.tick_params(axis='y', labelsize=tick_label_size)
    ax.tick_params(axis='z', labelsize=tick_label_size)

    # Set proportional axis
    max_range = np.array([x_coords, y_coords, z_coords]).max() * 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # Add legend
    ax.legend(loc='upper right', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add dipole moment information text
    dipole_text = f'Dipole Moment (Debye)\nX: {dipole["x"]:.4f}\nY: {dipole["y"]:.4f}\nZ: {dipole["z"]:.4f}\nTotal: {dipole["total"]:.4f}'
    ax.text2D(0.02, 0.98, dipole_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Make the view rotatable
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig, ax


def create_interactive_view(molecule_name, atoms, dipole, bonds):
    """Create an interactive view with sliders"""
    fig = plt.figure(figsize=(14, 10))

    # Create the main 3D model
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    # Initial view
    elev_init = 20
    azim_init = 45

    # Create slider axis
    ax_elev = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_azim = plt.axes([0.1, 0.05, 0.8, 0.03])

    # Create a slider
    slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=elev_init)
    slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=azim_init)

    # Update function
    def update(val):
        ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
        fig.canvas.draw_idle()

    slider_elev.on_changed(update)
    slider_azim.on_changed(update)

    # Plotting molecules and dipole moments
    visualize_molecule_with_dipole(molecule_name, atoms, dipole, bonds)

    plt.show()


def main():
    """main function"""
    # Create a sample file (if needed).
    create_example_file = True

    if create_example_file:
        example_content = """# Me4PACz
45

 C                    -0.36095  -0.01199   1.11329 
 C                     1.47658  -2.48474   0.5259 
 C                     2.45438  -3.41553   0.19861 
 C                     3.7409   -3.03843  -0.23708 
 C                     4.04211  -1.68343  -0.34342 
 C                     4.02432   1.69961  -0.3498 
 C                     3.70889   3.05181  -0.24869 
 C                     2.41845   3.41706   0.18537 
 C                     1.45048   2.47721   0.51615 
 N                     1.02624  -0.00567   0.68869 
 C                     1.7982   -1.13001   0.41693 
 C                     3.08106  -0.71957  -0.02174 
 C                     3.07349   0.72692  -0.02452 
 C                     1.78631   1.12554   0.41255 
 C                     4.71918   4.12031  -0.59547 
 C                     4.76234  -4.09764  -0.57964 
 C                    -1.3615   -0.01394  -0.05138 
 C                    -2.81377  -0.02413   0.43477 
 C                    -3.8158   -0.02252  -0.72776 
 P                    -5.538    -0.09935  -0.18874 
 O                    -5.71721   1.36821   0.47447 
 O                    -6.36688  -0.01833  -1.58501 
 O                    -5.96688  -1.22834   0.67193 
 H                    -0.51872  -0.88762   1.74954 
 H                    -0.52562   0.86014   1.7528 
 H                     0.49625  -2.81487   0.85292 
 H                     2.21687  -4.4728    0.27977 
 H                     5.02869  -1.37512  -0.67848 
 H                     5.01412   1.40044  -0.68363 
 H                     2.16978   4.47211   0.26213 
 H                     0.46646   2.7981    0.84137 
 H                     4.93001   4.76963   0.26066 
 H                     5.6675    3.68402  -0.91677 
 H                     4.3632    4.76501  -1.40581 
 H                     4.41305  -4.74934  -1.38728 
 H                     5.706    -3.65271  -0.90283 
 H                     4.97998  -4.74121   0.2791 
 H                    -1.18074   0.86589  -0.67853 
 H                    -1.17023  -0.88796  -0.68358 
 H                    -2.99213  -0.90516   1.06049 
 H                    -3.00003   0.84869   1.06915 
 H                    -3.70448   0.87004  -1.35047 
 H                    -3.66086  -0.88759  -1.37985 
 H                    -6.4627    1.4351    1.08364 
 H                    -7.05032  -0.69637  -1.64945 

# Dipole moment (field-independent basis, Debye):
    X=             -1.2388    Y=              0.9795    Z=             -0.6515  Tot=              1.7084"""

        with open('example_molecule.txt', 'w', encoding='utf-8') as f:
            f.write(example_content)
        print("Sample files have been created: example_molecule.txt")

    # Reading files
    filename = input("Please enter the file path and file name (e.g., C:\Users\example_molecule.txt, press Enter directly to use example_molecule.txt): ").strip()
    if not filename:
        filename = 'example_molecule.txt'

    try:
        # Parse file
        molecule_name, atoms, dipole = parse_xyz_dipole_file(filename)
        print(f"\nMolecular name: {molecule_name}")
        print(f"number of atoms: {len(atoms)}")
        print(
            f"dipole moment: X={dipole['x']:.4f}, Y={dipole['y']:.4f}, Z={dipole['z']:.4f}, Total={dipole['total']:.4f} Debye")

        # Calculation of chemical bonds
        bonds = calculate_bonds(atoms)
        print(f"Number of chemical bonds: {len(bonds)}")

        # Ask the user if they want to customize the font size.
        custom_font = input("\nDo you want to customize the font size?？(y/n，defaultn): ").strip().lower()

        if custom_font == 'y':
            try:
                axis_label_size = int(input("Please enter the font size for the axis labels. (default 14): ") or "14")
                tick_label_size = int(input("Please enter the font size for the coordinate axis ticks. (default 12): ") or "12")
                title_size = int(input("Please enter the title font size. (default 16): ") or "16")
                element_label_size = int(input("Please enter the element tag font size. (default 10): ") or "10")
            except ValueError:
                print("Invalid input, use default font size")
                axis_label_size, tick_label_size, title_size, element_label_size = 14, 12, 16, 10
        else:
            axis_label_size, tick_label_size, title_size, element_label_size = 14, 12, 16, 10

        # Visualization
        fig, ax = visualize_molecule_with_dipole(molecule_name, atoms, dipole, bonds,
                                                 axis_label_size=axis_label_size,
                                                 tick_label_size=tick_label_size,
                                                 title_size=title_size,
                                                 element_label_size=element_label_size)

        # Ask if you want to save the image
        save_image = input("\nSave image?？(y/n，default n): ").strip().lower()
        if save_image == 'y':
            # SVG format is used by default.
            output_filename = input("Please enter the output file name（default：molecule.svg）: ").strip()
            if not output_filename:
                output_filename = "molecule.svg"

            # Make sure the filename has the correct extension.
            if not output_filename.endswith(('.svg', '.png', '.jpg', '.jpeg', '.pdf')):
                output_filename += '.svg'

            # Determine the storage parameters based on the file extension
            if output_filename.endswith('.svg'):
                # SVG format - Vector graphics, infinitely scalable
                print(f"Saving SVG vector image to {output_filename}...")
                fig.savefig(output_filename, format='svg', bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"SVG Vector image saved: {output_filename}")
                print("Note: SVG format can be scaled infinitely, making it suitable for publishing and printing")
            elif output_filename.endswith('.pdf'):
                # PDF format - also a vector format
                print(f"Saving PDF vector image to {output_filename}...")
                fig.savefig(output_filename, format='pdf', bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"The PDF vector image has been saved: {output_filename}")
            else:
                # Other formats (PNG, JPG, etc.) - bitmap formats, using high DPI
                print(f"Saving high-resolution image to {output_filename}...")
                fig.savefig(output_filename, dpi=400, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"High-resolution images have been saved: {output_filename}")
                print(f"Image resolution: 4800x4000 Pixels (Super 4K)")

        # Add mouse interaction
        def on_move(event):
            if event.inaxes == ax and event.button:
                ax.view_init(elev=ax.elev + (event.ydata - on_move.last_y) * 0.5,
                             azim=ax.azim + (event.xdata - on_move.last_x) * 0.5)
                on_move.last_x = event.xdata
                on_move.last_y = event.ydata
                fig.canvas.draw_idle()

        def on_press(event):
            if event.inaxes == ax:
                on_move.last_x = event.xdata
                on_move.last_y = event.ydata

        on_move.last_x = 0
        on_move.last_y = 0

        fig.canvas.mpl_connect('motion_notify_event', on_move)
        fig.canvas.mpl_connect('button_press_event', on_press)

        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found '{filename}'")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":

    main()

