"""Streamlit 演示页：调用本地 API，展示点预测、预测区间与 SHAP 归因。

只做一件事——让别人在 30 秒内看懂这个系统在干什么。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st

API_URL = os.environ.get("HOUSEPRICE_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 10


def call(path: str, payload: dict | None = None) -> dict:
    url = f"{API_URL}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="ignore")
        raise RuntimeError(f"API 返回 {exc.code}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 {API_URL}，请先执行 `make serve`") from exc


st.set_page_config(page_title="HousePrice-AI", page_icon="🏠", layout="wide")
st.title("HousePrice-AI 房价预估")
st.caption("点预测 + 80% 预测区间 + SHAP 归因 | 模型：LightGBM / XGBoost / RF 交叉验证选优")

try:
    health = call("/health")
    st.success(
        f"服务在线 · 版本 {health['model_version']} · 算法 {health['algorithm']} · "
        f"特征数 {health['n_features']}"
    )
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("房屋信息")
    with st.form("predict"):
        col1, col2 = st.columns(2)
        overall_qual = col1.slider("整体质量 OverallQual", 1, 10, 7)
        year_built = col2.number_input("建造年份 YearBuilt", 1870, 2010, 2003)
        gr_liv_area = col1.number_input("地上居住面积 GrLivArea (sqft)", 300, 6000, 1710, step=50)
        total_bsmt = col2.number_input("地下室面积 TotalBsmtSF (sqft)", 0, 4000, 900, step=50)
        garage_cars = col1.slider("车库容量 GarageCars", 0, 4, 2)
        full_bath = col2.slider("完整卫生间 FullBath", 0, 4, 2)
        neighborhood = col1.selectbox("社区 Neighborhood", ["CollgCr", "NridgHt", "OldTown", "Edwards"])
        mszoning = col2.selectbox("区划 MSZoning", ["RL", "RM", "FV", "C (all)"])
        submitted = st.form_submit_button("开始预估", use_container_width=True)

features = {
    "OverallQual": overall_qual,
    "YearBuilt": year_built,
    "GrLivArea": gr_liv_area,
    "TotalBsmtSF": total_bsmt,
    "1stFlrSF": gr_liv_area,
    "GarageCars": garage_cars,
    "GarageArea": garage_cars * 250,
    "FullBath": full_bath,
    "Neighborhood": neighborhood,
    "MSZoning": mszoning,
}

with right:
    st.subheader("预估结果")
    if submitted:
        try:
            result = call("/predict", {"features": features})
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.metric("预估房价", f"${result['prediction']:,.0f}")
            interval = result["interval"]
            st.info(
                f"80% 预测区间：**${interval['low']:,.0f} — ${interval['high']:,.0f}**\n\n"
                f"区间宽度 ${interval['high'] - interval['low']:,.0f}，"
                "由 P10/P90 分位数回归直接给出，而非固定百分比误差。"
            )

            with st.expander("为什么给出这个价格？（SHAP 归因）", expanded=True):
                try:
                    explained = call("/explain", {"features": features})
                    rows = explained["items"][:8]
                    st.bar_chart(
                        {item["feature"]: item["contribution"] for item in rows},
                        horizontal=True,
                    )
                    st.caption("正值推高预测价，负值拉低预测价（单位：log 空间）。")
                except RuntimeError as exc:
                    st.warning(f"归因暂不可用：{exc}")
    else:
        st.info("左侧填写房屋信息后点击「开始预估」。")

st.divider()
st.caption(
    "接口文档：`/docs` · 模型版本切换：`POST /models/{version}/activate` · "
    "指标对比：`reports/metrics.md`"
)
