import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import re

# 元素的颜色映射
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

# 元素的原子半径（用于球的大小）
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

# 共价半径（用于判断成键）
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
    """解析包含xyz坐标和偶极矩的文件"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析分子名称
    molecule_name = lines[0].strip()
    if molecule_name.startswith('#'):
        molecule_name = molecule_name[1:].strip()

    # 解析原子数
    atom_count = int(lines[1].strip())

    # 解析原子坐标
    atoms = []
    for i in range(3, 3 + atom_count):
        parts = lines[i].strip().split()
        if len(parts) >= 4:
            element = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            atoms.append({'element': element, 'x': x, 'y': y, 'z': z})

    # 解析偶极矩
    dipole = {}
    for line in lines[3 + atom_count:]:
        if 'X=' in line and 'Y=' in line and 'Z=' in line:
            # 使用正则表达式提取数值
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
    """计算原子间的化学键"""
    bonds = []
    n_atoms = len(atoms)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            atom1 = atoms[i]
            atom2 = atoms[j]

            # 计算距离
            distance = np.sqrt(
                (atom2['x'] - atom1['x']) ** 2 +
                (atom2['y'] - atom1['y']) ** 2 +
                (atom2['z'] - atom1['z']) ** 2
            )

            # 判断是否成键
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
    可视化分子结构和偶极矩

    参数:
    - axis_label_size: 坐标轴标签字号
    - tick_label_size: 坐标轴刻度标签字号
    - title_size: 标题字号
    - element_label_size: 元素标签字号
    - fig_size: 图形尺寸（宽，高）

    注意：
    - 原子坐标单位：Å (埃)
    - 偶极矩单位：Debye
    - 偶极矩矢量仅用于显示方向，长度经过缩放以便于可视化
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # 提取原子坐标
    x_coords = [atom['x'] for atom in atoms]
    y_coords = [atom['y'] for atom in atoms]
    z_coords = [atom['z'] for atom in atoms]

    # 计算分子中心
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    center_z = np.mean(z_coords)

    # 中心化坐标
    x_coords = [x - center_x for x in x_coords]
    y_coords = [y - center_y for y in y_coords]
    z_coords = [z - center_z for z in z_coords]

    # 绘制化学键
    for bond in bonds:
        i, j = bond
        ax.plot([x_coords[i], x_coords[j]],
                [y_coords[i], y_coords[j]],
                [z_coords[i], z_coords[j]],
                'k-', linewidth=2, alpha=0.6)

    # 绘制原子
    for i, atom in enumerate(atoms):
        color = element_colors.get(atom['element'], 'gray')
        size = element_radii.get(atom['element'], 50)
        ax.scatter(x_coords[i], y_coords[i], z_coords[i],
                   c=color, s=size, edgecolors='black', linewidths=1, alpha=0.8)

        # 添加元素标签（可调整字号）
        ax.text(x_coords[i], y_coords[i], z_coords[i] + 0.3,
                atom['element'], fontsize=element_label_size, ha='center')

    # 绘制偶极矩矢量
    origin = [0, 0, 0]
    scale = 2  # 矢量缩放因子（仅用于可视化，不代表实际单位转换）

    # 注意：偶极矩的单位是Debye，而坐标轴的单位是Å
    # 这里的scale仅用于让矢量在图中清晰可见，不是单位转换

    # X分量（红色）
    if abs(dipole['x']) > 0.001:
        ax.quiver(origin[0], origin[1], origin[2],
                  dipole['x'] * scale, 0, 0,
                  color='red', arrow_length_ratio=0.3, linewidth=3,
                  label=f'X: {dipole["x"]:.3f} D')

    # Y分量（绿色）
    if abs(dipole['y']) > 0.001:
        ax.quiver(origin[0], origin[1], origin[2],
                  0, dipole['y'] * scale, 0,
                  color='green', arrow_length_ratio=0.3, linewidth=3,
                  label=f'Y: {dipole["y"]:.3f} D')

    # Z分量（蓝色）
    if abs(dipole['z']) > 0.001:
        ax.quiver(origin[0], origin[1], origin[2],
                  0, 0, dipole['z'] * scale,
                  color='blue', arrow_length_ratio=0.3, linewidth=3,
                  label=f'Z: {dipole["z"]:.3f} D')

    # 总偶极矩（黄色）
    ax.quiver(origin[0], origin[1], origin[2],
              dipole['x'] * scale, dipole['y'] * scale, dipole['z'] * scale,
              color='gold', arrow_length_ratio=0.3, linewidth=4,
              label=f'Total: {dipole["total"]:.3f} D')

    # 设置图形属性 - 可调整字号
    ax.set_xlabel('X (Å)', fontsize=axis_label_size, labelpad=10)
    ax.set_ylabel('Y (Å)', fontsize=axis_label_size, labelpad=10)
    ax.set_zlabel('Z (Å)', fontsize=axis_label_size, labelpad=10)
    ax.set_title(f'{molecule_name}\nDipole Moment Visualization', fontsize=title_size, pad=20)

    # 设置坐标轴刻度标签字号
    ax.tick_params(axis='x', labelsize=tick_label_size)
    ax.tick_params(axis='y', labelsize=tick_label_size)
    ax.tick_params(axis='z', labelsize=tick_label_size)

    # 设置等比例轴
    max_range = np.array([x_coords, y_coords, z_coords]).max() * 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # 添加图例
    ax.legend(loc='upper right', fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 添加偶极矩信息文本
    dipole_text = f'Dipole Moment (Debye)\nX: {dipole["x"]:.4f}\nY: {dipole["y"]:.4f}\nZ: {dipole["z"]:.4f}\nTotal: {dipole["total"]:.4f}'
    ax.text2D(0.02, 0.98, dipole_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 使视图可旋转
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig, ax


def create_interactive_view(molecule_name, atoms, dipole, bonds):
    """创建带滑块的交互式视图"""
    fig = plt.figure(figsize=(14, 10))

    # 创建主要的3D图
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    # 初始视角
    elev_init = 20
    azim_init = 45

    # 创建滑块轴
    ax_elev = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_azim = plt.axes([0.1, 0.05, 0.8, 0.03])

    # 创建滑块
    slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=elev_init)
    slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=azim_init)

    # 更新函数
    def update(val):
        ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
        fig.canvas.draw_idle()

    slider_elev.on_changed(update)
    slider_azim.on_changed(update)

    # 绘制分子和偶极矩
    visualize_molecule_with_dipole(molecule_name, atoms, dipole, bonds)

    plt.show()


def main():
    """主函数"""
    # 创建示例文件（如果需要）
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
        print("已创建示例文件: example_molecule.txt")

    # 读取文件
    filename = input("请输入文件名 (直接回车使用 example_molecule.txt): ").strip()
    if not filename:
        filename = 'example_molecule.txt'

    try:
        # 解析文件
        molecule_name, atoms, dipole = parse_xyz_dipole_file(filename)
        print(f"\n分子名称: {molecule_name}")
        print(f"原子数: {len(atoms)}")
        print(
            f"偶极矩: X={dipole['x']:.4f}, Y={dipole['y']:.4f}, Z={dipole['z']:.4f}, Total={dipole['total']:.4f} Debye")

        # 计算化学键
        bonds = calculate_bonds(atoms)
        print(f"化学键数: {len(bonds)}")

        # 询问用户是否要自定义字号
        custom_font = input("\n是否要自定义字号？(y/n，默认n): ").strip().lower()

        if custom_font == 'y':
            try:
                axis_label_size = int(input("请输入坐标轴标签字号 (默认14): ") or "14")
                tick_label_size = int(input("请输入坐标轴刻度字号 (默认12): ") or "12")
                title_size = int(input("请输入标题字号 (默认16): ") or "16")
                element_label_size = int(input("请输入元素标签字号 (默认10): ") or "10")
            except ValueError:
                print("输入无效，使用默认字号")
                axis_label_size, tick_label_size, title_size, element_label_size = 14, 12, 16, 10
        else:
            axis_label_size, tick_label_size, title_size, element_label_size = 14, 12, 16, 10

        # 可视化
        fig, ax = visualize_molecule_with_dipole(molecule_name, atoms, dipole, bonds,
                                                 axis_label_size=axis_label_size,
                                                 tick_label_size=tick_label_size,
                                                 title_size=title_size,
                                                 element_label_size=element_label_size)

        # 询问是否保存图片
        save_image = input("\n是否保存图片？(y/n，默认n): ").strip().lower()
        if save_image == 'y':
            # 默认使用SVG格式
            output_filename = input("请输入输出文件名（默认：molecule.svg）: ").strip()
            if not output_filename:
                output_filename = "molecule.svg"

            # 确保文件名有正确的扩展名
            if not output_filename.endswith(('.svg', '.png', '.jpg', '.jpeg', '.pdf')):
                output_filename += '.svg'

            # 根据扩展名确定保存参数
            if output_filename.endswith('.svg'):
                # SVG格式 - 矢量图形，可无限缩放
                print(f"正在保存SVG矢量图片到 {output_filename}...")
                fig.savefig(output_filename, format='svg', bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"SVG矢量图片已保存: {output_filename}")
                print("提示：SVG格式可无限缩放，适合出版和印刷")
            elif output_filename.endswith('.pdf'):
                # PDF格式 - 也是矢量格式
                print(f"正在保存PDF矢量图片到 {output_filename}...")
                fig.savefig(output_filename, format='pdf', bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"PDF矢量图片已保存: {output_filename}")
            else:
                # 其他格式（PNG, JPG等）- 位图格式，使用高DPI
                print(f"正在保存高分辨率图片到 {output_filename}...")
                fig.savefig(output_filename, dpi=400, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"高分辨率图片已保存: {output_filename}")
                print(f"图片分辨率: 4800x4000 像素 (超4K)")

        # 添加鼠标交互
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
        print(f"错误: 找不到文件 '{filename}'")
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()