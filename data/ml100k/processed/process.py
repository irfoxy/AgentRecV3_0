import os
import pandas as pd

# 输入和输出目录
input_dir = "../raw"
output_dir = "./"

# 定义生成的CSV文件路径
output_file = os.path.join(output_dir, "history.csv")

# 存储所有用户的交互记录
user_history = {}

# 遍历所有 uX.base 和 uX.test 文件
for file_name in os.listdir(input_dir):
    if file_name.endswith(".base"):
        # 读取数据文件
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # 按用户整理交互历史
        for _, row in df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            timestamp = row['timestamp']
            
            if user_id not in user_history:
                user_history[user_id] = {'history': [], 'rating': [], 'timestamp': []}
            
            # 将当前交互信息添加到该用户的历史记录中
            user_history[user_id]['history'].append(item_id)
            user_history[user_id]['rating'].append(rating)
            user_history[user_id]['timestamp'].append(timestamp)

# 处理每个用户的历史记录
history_data = []
for user_id, data in user_history.items():
    # 按时间戳排序交互历史
    sorted_indices = sorted(range(len(data['timestamp'])), key=lambda i: data['timestamp'][i])
    history = [data['history'][i] for i in sorted_indices]
    ratings = [data['rating'][i] for i in sorted_indices]
    
    # 寻找最后一个评分为4或5的物品
    ground_truth_item = None
    for i in range(len(ratings)-1, -1, -1):  # 从后往前遍历
        if ratings[i] in [4, 5]:
            ground_truth_item = int(history[i])  # 转换为整数类型
            break
    
    # 如果没有找到ground_truth，则舍弃该用户
    if ground_truth_item is None:
        continue
    
    # 删除该物品后续的所有物品
    ground_truth_index = history.index(ground_truth_item)
    history = history[:ground_truth_index+1]  # 保留到 ground_truth_item 为止
    ratings = ratings[:ground_truth_index+1]  # 同样调整评分列表

    # 将数据加入最终输出
    history_data.append([user_id, ' '.join(map(str, history)), ' '.join(map(str, ratings)), int(ground_truth_item)])

# 将数据写入到 CSV 文件
history_df = pd.DataFrame(history_data, columns=['user_id', 'history', 'rating', 'ground_truth'])
history_df.to_csv(output_file, index=False)

print(f"history.csv 已成功生成，路径为: {output_file}")
