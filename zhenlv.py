import pandas as pd
from datetime import datetime, timedelta

# 设置文件路径
file_path = r'C:\Users\18155\Desktop\gather.csv'
output_file_path = r'C:\Users\18155\Desktop\1.csv'  # 处理后的文件路径

# 视频开始时间和帧率
start_time = datetime.strptime('09:34:43', '%H:%M:%S')  # 修正格式字符串
frames_per_second = 25

# 读取CSV文件，没有表头
df = pd.read_csv(file_path, header=None)

# 定义转换函数：帧数标识转换为时间戳
def frame_to_time(frame_str):
    if isinstance(frame_str, str) and '.txt' in frame_str:
        frame_num = int(frame_str.replace('.txt', '')) - 1  # 帧数是从1开始的，时间计算从0开始
        return (start_time + timedelta(seconds=frame_num / frames_per_second)).strftime('%H:%M:%S')
    else:
        return frame_str

# 应用转换函数到第一列（A列）
df[0] = df[0].apply(frame_to_time)

# 保存处理后的数据到新的CSV文件
df.to_csv(output_file_path, index=False, header=False)
