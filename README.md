# Home Credit Default Risk - Baseline

这是 Kaggle 比赛 "Home Credit Default Risk" 的一个基础 baseline 模型。

## 项目结构

```
home-credit-default-risk/       # 数据目录
├── application_train.csv       # 训练集主表
├── application_test.csv        # 测试集主表
├── bureau.csv                  # 信用局数据
├── bureau_balance.csv          # 信用局月度余额
├── credit_card_balance.csv     # 信用卡余额
├── installments_payments.csv   # 分期付款历史
├── POS_CASH_balance.csv        # POS 和现金贷款余额
├── previous_application.csv    # 历史申请记录
└── HomeCredit_columns_description.csv  # 列描述

home-credit-baseline.py         # 主程序
requirements.txt                # 依赖包
submission.csv                  # 生成的提交文件
```

## 功能特点

### 1. 数据处理
- 加载所有数据表（主表 + 6个辅助表）
- 处理异常值（如 DAYS_EMPLOYED 的异常值）
- 缺失值处理

### 2. 特征工程
- **主表特征**：
  - 收入相关比率（信贷/收入比、年金/收入比等）
  - 年龄和工作年限
  - 家庭相关特征（人均收入、子女比例）
  - 文档数量统计
  - 外部评分统计（均值、标准差、乘积）

- **Bureau 特征**：
  - 历史信贷的各种聚合统计
  - 逾期情况统计
  - 活跃和关闭账户数量

- **Previous Application 特征**：
  - 历史申请的聚合统计
  - 批准和拒绝申请数量

- **Installments 特征**：
  - 还款差异和比率
  - 逾期天数统计

- **Credit Card 特征**：
  - 信用卡使用情况
  - 取款和还款统计
  - DPD（逾期天数）统计

- **POS Cash 特征**：
  - POS 和现金贷款余额统计
  - 分期付款情况

### 3. 模型
- 使用 LightGBM 梯度提升模型
- 评估指标：ROC AUC
- 包含早停机制和交叉验证
- 输出特征重要性

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行 Baseline

```bash
python home-credit-baseline.py
```

程序会自动：
1. 加载并处理所有数据
2. 进行特征工程
3. 训练 LightGBM 模型
4. 生成预测结果并保存到 `submission.csv`

### 3. 提交结果

生成的 `submission.csv` 文件可以直接提交到 Kaggle。

## 预期性能

这个 baseline 模型预期可以达到：
- 验证集 AUC: 0.75-0.77
- Public Leaderboard AUC: 0.74-0.76

## 改进方向

1. **特征工程优化**：
   - 添加更多的交互特征
   - 更精细的时间窗口特征
   - 使用 target encoding

2. **模型优化**：
   - 超参数调优
   - 使用 K-fold 交叉验证
   - 模型融合（LightGBM + XGBoost + CatBoost）

3. **数据处理**：
   - 更复杂的缺失值填充策略
   - 异常值处理
   - 特征选择

4. **深度特征提取**：
   - 从 bureau_balance 提取更多时序特征
   - 分析客户的还款行为模式

## 注意事项

- 数据量较大，处理可能需要几分钟时间
- 建议至少 8GB 内存
- 如果内存不足，可以考虑减少特征或使用采样

## 参考资料

- [Kaggle 比赛页面](https://www.kaggle.com/c/home-credit-default-risk)
- [LightGBM 文档](https://lightgbm.readthedocs.io/)

