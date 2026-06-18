# import pandas as pd

# df = pd.read_csv("merged_br_exp.csv")
# features = ["br_avg", "br_std", "act_level", "step_count", "latitude", "longitude",
#             "pm_dose_hr", "pm_dose_rate", "pm2_5", "temperature", "humidity"]
# # missing_data = (df[features] == -1).any(axis=1) 

# rate = 0.4
# bad_patients = []
# # missing_rate = missing_data.groupby(df["patient_id"]).mean()
# # id_list = missing_rate[missing_rate >= rate].index
# # df_filtered = df[~df["patient_id"].isin(id_list)]
# for pid, g in df.groupby("patient_id"):
#     # column-wise missing rate
#     col_missing_rate = (g[features] == -1).mean()

#     # drop patient if ANY feature exceeds threshold
#     if (col_missing_rate >= rate).any():
#         bad_patients.append(pid)

# # -------------------------
# # Filter dataframe
# # -------------------------
# df_filtered = df[~df["patient_id"].isin(bad_patients)]

# for (cid, season), df_patient in df_filtered.groupby(["patient_id", "season"]):
#     df_patient.to_excel(f"{cid}_{season}.xlsx", index=False)

# # # # ##

import pandas as pd
import numpy as np
from pathlib import Path

# =====================
# 参数区（按需修改）
# =====================
input_excel = "merged_br_exp.csv"        # 原始 Excel
id_col = "patient_id"
season_col = "season"
missing_value = -1
threshold = 0.4                   # 20%

# value_cols = ['br_avg', 'br_std', 'act_level', 'step_count']


# =====================
# 读取 Excel
# =====================
df = pd.read_csv(input_excel)

# =====================
# 按 patient_id + season 分组
# =====================
grouped = df.groupby([id_col, season_col])

kept, dropped = 0, 0

for (pid, season), g in grouped:
    # ---------------------
    # 计算 -1 的占比
    # ---------------------
    value_cols = [col for col in g.columns if col not in [id_col, season_col, "timestamp"]]

    # ---------------------
    # 判断是否删除
    # ---------------------
    col_missing_rate = (g[value_cols] == missing_value).mean()
    if (col_missing_rate > threshold).any():
        dropped += 1
        continue

    for col in g.columns:
        if col=='patient_id' or col== 'season':
            continue
        if col=='timestamp':
            # 转换为 datetime 类型
            timestamp_series = pd.to_datetime(g[col])
            year = timestamp_series.dt.year
            month = timestamp_series.dt.month
            day = timestamp_series.dt.day
            year_info_cos = np.cos(2 * 3.14159 * (year-2000) / 12)
            year_info_sin = np.sin(2 * 3.14159 * (year-2000) / 12)
            month_info_sin = np.sin(2 * 3.14159 * month / 12)
            month_info_cos = np.cos(2 * 3.14159 * month / 12)
            day_info_sin = np.sin(2 * 3.14159 * day / 31)
            day_info_cos = np.cos(2 * 3.14159 * day / 31)
            g['year_info_sin'] = year_info_sin
            g['year_info_cos'] = year_info_cos
            g['month_info_sin'] = month_info_sin
            g['month_info_cos'] = month_info_cos
            g['day_info_sin'] = day_info_sin
            g['day_info_cos'] = day_info_cos
            g.drop(columns=['timestamp'], inplace=True)
            continue
        s = g[col]
        s[s == -1] = pd.NA
        s = s.ffill().bfill()  # 先前向填充，再后向填充以处理首行NaN
        g[col] = s
    g.drop(columns=['patient_id'], inplace=True)
    g.drop(columns=['season'], inplace=True)
    print(len(g))

    # ---------------------
    # 保存 Excel
    # ---------------------
    filename = f"{pid}_{season}.xlsx"
    out_path = Path(filename)
    g.to_excel(out_path, index=False)
    kept += 1

print(f"✅ 保留表格数: {kept}")
print(f"🗑️ 删除表格数: {dropped}")
