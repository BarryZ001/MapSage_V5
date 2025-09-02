# 文件名: final_correct_config.py
# 这是最可靠的推理配置文件版本

# --- 继承官方SegFormer-B2模型配置 ---
_base_ = [
    './_base_/models/segformer_mit-b2.py',
]

# --- 定义与我们任务相关的参数 ---
model = dict(
    # 唯一需要修改的模型部分：将解码头的类别数改为7
    decode_head=dict(
        num_classes=7
    ),

    # 定义滑窗推理，这是模型对象的一部分
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))
)

# 类别名称（用于可视化）
classes = [
    'background',    # 背景
    'building',      # 建筑
    'road',          # 道路
    'water',         # 水体
    'barren',        # 贫瘠土地
    'forest',        # 森林
    'agricultural'   # 农业
]

# 调色板（RGB格式）
palette = [
    [255, 255, 255],  # 背景 - 白色
    [255, 0, 0],      # 建筑 - 红色
    [255, 255, 0],    # 道路 - 黄色
    [0, 0, 255],      # 水体 - 蓝色
    [159, 129, 183],  # 贫瘠土地 - 紫色
    [0, 255, 0],      # 森林 - 绿色
    [255, 195, 128]   # 农业 - 橙色
]