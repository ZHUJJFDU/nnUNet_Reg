import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation, disk
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from PIL import Image
import matplotlib
from skimage.segmentation import find_boundaries
import json

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'Arial', 'sans-serif']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_image(image_path):
    # 加载 NIfTI 图像
    img = nib.load(image_path)
    img_data = img.get_fdata()
    return img_data, img.affine

def calculate_volume(segmentation, label_value):
    # 计算指定类别的体积
    volume = np.sum(segmentation == label_value)
    return volume

def find_max_area_slice(segmentation, label_value):
    # 找到指定类别在所有切片中面积最大的那个切片
    max_area = 0
    max_slice_index = 0
    max_diameter = 0
    max_slice = None
    max_region_props = None
    
    # 遍历每个切片
    for i in range(segmentation.shape[0]):
        slice_2d = segmentation[i, :, :]
        
        # 找到该切片中属于指定类别的区域
        binary_slice = (slice_2d == label_value)
        if not np.any(binary_slice):
            continue
            
        labeled_slice = label(binary_slice)
        regions = regionprops(labeled_slice)
        
        # 计算该切片上所有区域的总面积
        total_area = sum(region.area for region in regions)
        
        # 如果面积更大，则更新最大面积和索引
        if total_area > max_area:
            max_area = total_area
            max_slice_index = i
            max_slice = binary_slice
            
            # 找到最大的区域并获取其直径
            max_region = max(regions, key=lambda x: x.area, default=None)
            if max_region:
                max_diameter = max_region.equivalent_diameter
                max_region_props = max_region
    
    return max_slice_index, max_area, max_diameter, max_slice, max_region_props

def perform_axis_analysis(region_props):
    """
    对区域进行主轴分析，计算长轴、短轴和方向
    """
    if region_props is None:
        return 0, 0, 0, (0, 0)
    
    # 提取区域的中心坐标
    centroid = region_props.centroid
    
    # 计算区域的惯性张量
    inertia_tensor = region_props.inertia_tensor
    
    # 通过奇异值分解(SVD)找到主方向
    _, _, vh = np.linalg.svd(inertia_tensor)
    
    # 获取长轴和短轴长度
    major_axis_length = region_props.major_axis_length
    minor_axis_length = region_props.minor_axis_length
    
    # 主方向是惯性张量的第一个特征向量
    orientation = np.arctan2(vh[0, 1], vh[0, 0])
    
    return major_axis_length, minor_axis_length, orientation, centroid

def calculate_wall_thickness(segmentation, max_slice_index, cavity_label=2, wall_label=3):
    # 在最大切片上计算肺大泡壁的厚度
    max_slice = segmentation[max_slice_index, :, :]
    
    # 获取肺大泡内部和壁的二值掩码
    cavity_mask = (max_slice == cavity_label)
    wall_mask = (max_slice == wall_label)
    
    # 如果没有壁或内部，返回0
    if not np.any(cavity_mask) or not np.any(wall_mask):
        return 0, max_slice, None, 0
    
    # 计算从内部到壁的距离场
    distance = distance_transform_edt(~cavity_mask)
    
    # 获取壁上的距离值（即壁的厚度）
    wall_thickness_values = distance[wall_mask]
    
    # 计算平均厚度
    avg_wall_thickness = np.mean(wall_thickness_values) if len(wall_thickness_values) > 0 else 0
    
    # 创建壁厚度映射，保留壁区域的距离值，其他区域设为0
    thickness_map = np.zeros_like(distance)
    thickness_map[wall_mask] = distance[wall_mask]
    
    # 计算壁的完整性
    wall_completeness = calculate_wall_completeness(cavity_mask, wall_mask)
    
    return avg_wall_thickness, max_slice, thickness_map, wall_completeness

def crop_to_content(slice_data, mask, padding=20):
    """裁剪图像到内容区域，添加适当的填充"""
    # 找到掩码中的非零点
    y, x = np.where(mask)
    if len(y) == 0 or len(x) == 0:  # 如果掩码为空，返回原始切片
        return slice_data
    
    # 计算边界框
    y_min, y_max = np.min(y), np.max(y)
    x_min, x_max = np.min(x), np.max(x)
    
    # 添加填充
    y_min = max(0, y_min - padding)
    y_max = min(slice_data.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(slice_data.shape[1], x_max + padding)
    
    # 裁剪图像
    return slice_data[y_min:y_max, x_min:x_max], (y_min, y_max, x_min, x_max)

def generate_thickness_heatmap(original_slice, thickness_map, output_path, title):
    """
    生成壁厚度热图，直观显示壁厚的局部变化
    """
    # 找到非零区域用于裁剪
    mask = thickness_map > 0
    expanded_mask = binary_dilation(mask, disk(10))
    cropped_slice, crop_coords = crop_to_content(original_slice, expanded_mask, padding=40)
    cropped_thickness = thickness_map[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
    
    # 创建图像
    plt.figure(figsize=(12, 10))
    
    # 绘制裁剪后的原始切片作为背景
    plt.imshow(cropped_slice, cmap='gray')
    
    # 在壁区域上叠加热图
    masked_thickness = np.ma.masked_where(cropped_thickness == 0, cropped_thickness)
    plt.imshow(masked_thickness, cmap='jet', alpha=0.7)
    
    plt.colorbar(label='壁厚度 (像素)')
    plt.title(title, fontsize=16)  # 调整标题字体大小
    plt.axis('off')
    
    # 保存热图
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_cavity_with_axes(slice_data, cavity_mask, wall_mask, major_axis, minor_axis, orientation, centroid, output_path, title):
    """
    可视化肺大泡及其主轴
    
    参数:
        slice_data: 原始切片数据
        cavity_mask: 肺大泡腔体掩码
        wall_mask: 肺大泡壁掩码
        major_axis, minor_axis, orientation, centroid: 主轴分析结果
        output_path: 输出图像路径
        title: 图像标题
    """
    # 找到非零区域用于裁剪
    expanded_mask = binary_dilation(cavity_mask, disk(10))
    cropped_slice, crop_coords = crop_to_content(slice_data, expanded_mask, padding=40)
    cropped_cavity_mask = cavity_mask[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
    
    # 调整中心点坐标以适应裁剪
    adjusted_centroid = (
        centroid[0] - crop_coords[0],  # y坐标调整
        centroid[1] - crop_coords[2]   # x坐标调整
    )
    
    # 创建图像
    plt.figure(figsize=(10, 8))
    
    # 显示裁剪后的原始切片
    plt.imshow(cropped_slice, cmap='gray')
    
    # 叠加肺大泡掩码
    masked_cavity = np.ma.masked_where(~cropped_cavity_mask, cropped_cavity_mask)
    plt.imshow(masked_cavity, cmap='cool', alpha=0.3)
    
    # 绘制椭圆表示主轴和短轴
    ellipse = Ellipse(
        xy=(adjusted_centroid[1], adjusted_centroid[0]),  # 注意坐标系转换
        width=major_axis,
        height=minor_axis,
        angle=np.degrees(orientation),
        fill=False,
        edgecolor='red',
        linewidth=2
    )
    plt.gca().add_patch(ellipse)
    
    # 绘制主轴和短轴
    cos_angle = np.cos(orientation)
    sin_angle = np.sin(orientation)
    
    # 主轴 - 注意坐标系转换
    x_center, y_center = adjusted_centroid[1], adjusted_centroid[0]  # 转换为绘图坐标系
    major_x = major_axis/2 * cos_angle
    major_y = major_axis/2 * sin_angle
    plt.plot(
        [x_center - major_x, x_center + major_x], 
        [y_center - major_y, y_center + major_y], 
        'r-', linewidth=2, label='主轴'
    )
    
    # 短轴 - 保证与主轴垂直
    minor_x = minor_axis/2 * -sin_angle  # 注意负号确保垂直
    minor_y = minor_axis/2 * cos_angle
    plt.plot(
        [x_center - minor_x, x_center + minor_x], 
        [y_center - minor_y, y_center + minor_y], 
        'y-', linewidth=2, label='短轴'
    )
    
    # 处理缺失壁区域的可视化
    if wall_mask is not None:
        # 裁剪wall_mask以匹配cropped_cavity_mask
        cropped_wall_mask = wall_mask[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
        
        # 获取腔体边界
        cavity_boundary = find_boundaries(cropped_cavity_mask)
        
        # 识别缺失的壁区域
        expanded_wall = binary_dilation(cropped_wall_mask, disk(2))
        missing_wall = cavity_boundary & ~expanded_wall
        
        # 可视化缺失区域
        plt.contour(missing_wall, colors='magenta', linewidths=2, 
                   levels=[0.5], label='缺失壁区域')
    
    plt.title(title, fontsize=16)  # 调整标题字体大小
    plt.legend(loc='upper right')
    plt.axis('off')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def calculate_wall_completeness(cavity_mask, wall_mask):
    """计算壁的完整性"""
    # 获取腔体的边界
    cavity_boundary = find_boundaries(cavity_mask)
    
    # 理想情况下，边界上每个点都应该有相邻的壁
    # 扩展壁区域，检查它是否覆盖了腔体边界
    expanded_wall = binary_dilation(wall_mask, disk(2))
    
    # 计算边界点总数和被壁覆盖的边界点数量
    total_boundary_points = np.sum(cavity_boundary)
    covered_boundary_points = np.sum(cavity_boundary & expanded_wall)
    
    # 计算覆盖率
    coverage_percentage = (covered_boundary_points / total_boundary_points * 100) if total_boundary_points > 0 else 0
    
    return coverage_percentage

def analyze_wall_thickness_distribution(thickness_map):
    """分析壁厚度的分布情况"""
    # 排除零值（非壁区域）
    thickness_values = thickness_map[thickness_map > 0]
    
    if len(thickness_values) == 0:
        return {"min": 0, "max": 0, "median": 0, "std": 0, "percentiles": [0, 0, 0]}
    
    # 计算基本统计量
    min_thickness = np.min(thickness_values)
    max_thickness = np.max(thickness_values)
    median_thickness = np.median(thickness_values)
    std_thickness = np.std(thickness_values)
    
    # 计算百分位数 (25%, 50%, 75%)
    percentiles = np.percentile(thickness_values, [25, 50, 75])
    
    return {
        "min": min_thickness,
        "max": max_thickness,
        "median": median_thickness,
        "std": std_thickness,
        "percentiles": percentiles
    }

def process_images(folder_path, folder_path1,output_dir=None):
    # 如果没有指定输出目录，则使用输入目录
    if output_dir is None:
        output_dir = os.path.join(folder_path, "results")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件夹中所有 .nii.gz 文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
    
    # 创建一个字典来存储所有图像的壁厚度结果
    thickness_results = {}
    
    # 处理每个图像文件
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"处理文件: {image_path}...")
        
        # 提取文件名（不包含扩展名）作为键
        base_name = os.path.splitext(os.path.splitext(image_file)[0])[0]
        
        # 加载图像
        segmentation, affine = load_image(image_path)
        
        # 找到肺大泡内部(标签2)最大的z轴切片
        max_slice_index, max_area, max_diameter, max_cavity_slice, max_region_props = find_max_area_slice(segmentation, label_value=2)
        
        # 进行主轴分析
        major_axis, minor_axis, orientation, centroid = perform_axis_analysis(max_region_props)
        
        # 计算肺大泡壁的厚度
        wall_thickness, max_slice, thickness_map, wall_completeness = calculate_wall_thickness(segmentation, max_slice_index)
        
        # 计算体素大小（用于转换成实际物理尺寸，如毫米）
        voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        
        # 将像素单位转换为物理单位（如毫米）
        physical_wall_thickness = wall_thickness * voxel_sizes[1]  # 平均壁厚
        
        # 存储壁厚度结果到字典
        thickness_results[base_name] = {
            "bulla_thickness": round(float(physical_wall_thickness), 2)
        }
        
        # 生成可视化图像
        if thickness_map is not None:
            # 热图输出路径
            heatmap_path = os.path.join(output_dir, f"{base_name}_wall_thickness_heatmap.png")
            
            # 生成壁厚度热图
            generate_thickness_heatmap(segmentation[max_slice_index], thickness_map,
                                    heatmap_path, f"肺大泡壁厚度热图 (切片 #{max_slice_index})")
            
            thickness_stats = analyze_wall_thickness_distribution(thickness_map)
            print(f"肺大泡壁厚度统计:")
            print(f" - 最小厚度: {thickness_stats['min'] * voxel_sizes[1]:.2f} mm")
            print(f" - 最大厚度: {thickness_stats['max'] * voxel_sizes[1]:.2f} mm")
            print(f" - 中位数厚度: {thickness_stats['median'] * voxel_sizes[1]:.2f} mm")
            print(f" - 标准差: {thickness_stats['std'] * voxel_sizes[1]:.2f} mm")
            print(f" - 25/50/75百分位厚度: {[p * voxel_sizes[1] for p in thickness_stats['percentiles']]}")
        
        # 获取肺大泡壁掩码
        wall_mask = (segmentation[max_slice_index] == 3)  # 假设标签3是壁

        # 可视化肺大泡及其主轴
        axes_path = os.path.join(output_dir, f"{base_name}_cavity_axes.png")
        visualize_cavity_with_axes(segmentation[max_slice_index], max_cavity_slice,
                                  wall_mask,  # 传递wall_mask参数
                                  major_axis, minor_axis, orientation, centroid,
                                  axes_path, f"肺大泡主轴分析 (切片 #{max_slice_index})")
        
        # 输出结果信息
        print("-" * 50)
        print(f"图像: {os.path.basename(image_path)}")
        print(f"最大切片索引: {max_slice_index}")
        print(f"肺大泡面积 (像素): {max_area:.2f}")
        print(f"肺大泡等效直径 (mm): {max_diameter * voxel_sizes[1]:.2f}")
        print(f"肺大泡主轴长度 (mm): {major_axis * voxel_sizes[1]:.2f}")
        print(f"肺大泡短轴长度 (mm): {minor_axis * voxel_sizes[1]:.2f}")
        print(f"肺大泡长宽比: {major_axis/minor_axis if minor_axis > 0 else 0:.2f}")
        print(f"肺大泡壁平均厚度 (mm): {physical_wall_thickness:.2f}")
        print(f"肺大泡壁完整性: {wall_completeness:.2f}%")
        print(f"可视化输出:")
        print(f" - 壁厚度热图: {os.path.basename(heatmap_path) if thickness_map is not None else '无'}")
        print(f" - 主轴分析图: {os.path.basename(axes_path)}")
        print("-" * 50)
    
    # 将结果保存为 JSON 文件
    json_output_path = os.path.join(folder_path1, "regression_values.json")
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(thickness_results, json_file, ensure_ascii=False, indent=4)
    
    # 输出汇总信息
    print("\n=== 处理完成 ===")
    print(f"共处理了 {len(image_files)} 个文件")
    print(f"结果保存在: {output_dir}")
    print(f"壁厚度结果已保存为 JSON 文件: {json_output_path}")



def main():
    # 设置文件夹路径
    folder_path = r"C:\Users\Administrator\Desktop\nnUNet-master\DATASET\nnUNet_raw\Dataset103_quan\labelsTr"
    folder_path1 = r"C:\Users\Administrator\Desktop\nnUNet-master\DATASET\nnUNet_raw\Dataset103_quan"
    # 设置输出目录
    output_dir = os.path.join(folder_path, "enhanced_analysis")
    
    # 处理文件夹中的图像数据
    process_images(folder_path, folder_path1,output_dir)

if __name__ == '__main__':
    main()