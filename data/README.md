# 数据集说明

本项目使用 **Ames Housing**（Kaggle "House Prices - Advanced Regression Techniques" 同源数据），
通过 OpenML 公开接口获取，无需 Kaggle 账号。

- 规模：1460 行 × 80 列，目标列 `SalePrice`
- 特征：23 个名义类别 + 23 个有序类别 + 34 个数值
- 缺失：`PoolQC`/`MiscFeature`/`Alley`/`Fence` 缺失率超过 90%

## 获取方式

```bash
python -c "from houseprice.data.loader import fetch_ames; print(fetch_ames().shape)"
```

首次运行会从 OpenML 拉取并缓存为 `data/raw/ames.csv`；之后离线可用，保证实验可复现。

## 为什么不使用 Boston 数据集

1. **伦理问题**：`B`（城镇黑人比例）这一特征带有种族指向性，scikit-learn 已在 1.2 版本彻底移除 `load_boston`；
2. **信息量过低**：506 条样本、13 个纯数值特征、零缺失值，无法体现数据清洗与特征工程能力；
3. **目标被截断**：`MEDV` 在 50（千美元）处被人工截断，用 `clip` 掩盖会掩盖真实的建模缺陷。

Ames 的类别特征、缺失结构与样本量，足以支撑一套完整的工程实践。
