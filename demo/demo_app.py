import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("AI-Driven CRE Risk Analytics Demo")

# -------------------------
# 1. 模拟数据
# -------------------------
np.random.seed(42)

n = 50

data = pd.DataFrame({
    "property_id": range(n),
    "occupancy": np.random.uniform(0.6, 1.0, n),
    "rent": np.random.uniform(20, 80, n),
    "interest_rate": np.random.uniform(3, 8, n),
    "region": np.random.choice(["A", "B", "C"], n)
})

# -------------------------
# 2. 用户输入（宏观因素）
# -------------------------
st.sidebar.header("Macro Scenario")

interest_input = st.sidebar.slider("Interest Rate (%)", 2.0, 10.0, 5.0)
demand_input = st.sidebar.slider("Market Demand", 0.5, 1.5, 1.0)

# 更新数据
data["interest_rate"] = interest_input
data["demand"] = demand_input

# -------------------------
# 3. Deep Learning（简化版 → Logistic）
# -------------------------
features = data[["occupancy", "rent", "interest_rate", "demand"]]
scaler = StandardScaler()
X = scaler.fit_transform(features)

y = (data["occupancy"] < 0.75).astype(int)

model = LogisticRegression()
model.fit(X, y)

risk_scores = model.predict_proba(X)[:, 1]
data["risk_score"] = risk_scores

# -------------------------
# 4. GNN（简化版 → Graph传播）
# -------------------------
G = nx.Graph()

for i in range(n):
    G.add_node(i, risk=risk_scores[i])

# 同区域连接
for i in range(n):
    for j in range(i+1, n):
        if data.loc[i, "region"] == data.loc[j, "region"]:
            G.add_edge(i, j)

# 简单传播（平均邻居风险）
for node in G.nodes:
    neighbors = list(G.neighbors(node))
    if neighbors:
        neighbor_risk = np.mean([risk_scores[n] for n in neighbors])
        data.loc[node, "risk_score"] = 0.7 * data.loc[node, "risk_score"] + 0.3 * neighbor_risk

# -------------------------
# 5. 输出结果
# -------------------------
st.subheader("Property Risk Scores")
st.dataframe(data[["property_id", "region", "risk_score"]])

# -------------------------
# 6. 可视化
# -------------------------
st.subheader("Risk Distribution")

fig, ax = plt.subplots()
ax.hist(data["risk_score"], bins=10)
ax.set_title("Risk Score Distribution")
st.pyplot(fig)

# -------------------------
# 7. 简单LLM模拟（解释）
# -------------------------
st.subheader("AI Risk Insight")

avg_risk = data["risk_score"].mean()

if avg_risk > 0.6:
    insight = "Portfolio risk is elevated due to high interest rates and weak demand."
elif avg_risk > 0.4:
    insight = "Moderate risk detected. Some regions show clustering risk patterns."
else:
    insight = "Portfolio risk is relatively low under current macro conditions."

st.write(insight)
