import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import io
# 1. メモリ上にバイナリデータを保存するためのバッファを作成
buf = io.BytesIO()


# --- UI Setup ---
st.set_page_config(page_title="Composition of maximal operators ", layout="wide")
st.title("Compositions of maximal operators behaviour:\n $\\left( M_{\\alpha}\\circ M_{\\beta}\\right) \\left[ \\chi_{[0,1]}\\right](x)$ and $\\left( M_{\\beta}\\circ M_{\\alpha}\\right)\\left[ \\chi_{[0,1]}\\right](x)$")


# --- 1. 数学的計算 ---
def get_peak_safe(a, b):
    try:
        # 数値的に不安定な極小値を避ける
        a = max(a, 1e-6)
        b = max(b, 1e-6)
        if b < 1e-5:
            x0 = np.exp(a / (1 - a))
            y0 = x0**(a - 1) * (1 + np.log(x0))
        else:
            num = (1 - a) * (1 - b)
            den = 1 - (a + b)
            base = num / den
            x0 = base**(1 / b)
            y0 = x0**(a - 1) * (1 + (x0**b - 1) / b)
        return float(np.real(x0)), float(np.real(y0))
    except:
        return 1.1, 1.0

with st.sidebar:
    st.header("Parameter Constraints")
    S = st.slider("Fixed Sum $\\alpha + \\beta$", 0.02, 0.99, 0.50)
    
    # エラー回避：min < max を保証するために 0.0001 のバッファを入れる
    alpha_max = max(S - 0.001, 0.0101) 
    alpha = st.slider("$\\alpha$", 0.001, alpha_max, S / 2)
    
    beta = round(S - alpha, 4)
    # 負の値を防ぐ
    beta = max(beta, 0.0)
    
    x0_ab, y0_ab = get_peak_safe(alpha, beta)
    x0_ba, y0_ba = get_peak_safe(beta, alpha)
    
    st.info(f"$\\alpha+\\beta$: {S:.2f} | $\\alpha$: {alpha:.3f}, $\\beta$: {beta:.3f}")
    
    st.header("View Settings")
    show_m2 = st.checkbox("Show M^2 = M◦M", value=False)
    
    # ビューレンジの計算
    peak_max = max(x0_ab, x0_ba)
    default_view = float(peak_max * 1.5)
    # スライダーの最小値を2.0、最大値を1000に固定して安定させる
    x_view = st.slider("View Range (±x)", 2.0, 1000.0, min(max(default_view, 5.0), 1000.0))

def composition_func(x, A, B):
    val = np.ones_like(x, dtype=float)
    idx_pos = x > 1
    xx_p = x[idx_pos]
    if B < 1e-5:
        val[idx_pos] = xx_p**(A-1) * (1 + np.log(xx_p))
    else:
        val[idx_pos] = xx_p**(A-1) * (1 + (xx_p**B - 1) / B)
    idx_neg = x < 0
    xx_n = 1 - x[idx_neg]
    if B < 1e-5:
        val[idx_neg] = xx_n**(A-1) * (1 + np.log(xx_n))
    else:
        val[idx_neg] = xx_n**(A-1) * (1 + (xx_n**B - 1) / B)
    return val

def MM_iterated_correct(x):
    val = np.ones_like(x, dtype=float)
    mask_pos = x > 1
    val[mask_pos] = (1 + np.log(x[mask_pos])) / x[mask_pos]
    mask_neg = x < 0
    val[mask_neg] = (1 + np.log(1 - x[mask_neg])) / (1 - x[mask_neg])
    return val

# --- 2. 描画 ---
x_plot = np.linspace(-x_view, x_view, 3000)
y_ab = composition_func(x_plot, alpha, beta)
y_ba = composition_func(x_plot, beta, alpha)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_plot, y_ab, lw=2.5, color="royalblue", label=rf"$M_{{{alpha:.3f}}} \circ M_{{{beta:.3f}}}$")
ax.plot(x_plot, y_ba, lw=2.5, color="crimson", ls="--", label=rf"$M_{{{beta:.3f}}} \circ M_{{{alpha:.3f}}}$")

if show_m2:
    ax.plot(x_plot, MM_iterated_correct(x_plot), lw=1.5, color="#333333", alpha=0.7, label="$M^2$")

ax.fill_between(x_plot, np.where((x_plot>=0)&(x_plot<=1), 1, 0), color="gray", alpha=0.1)

# ピークのプロット
ax.plot([x0_ab, 1-x0_ab], [y0_ab, y0_ab], 'bo', markersize=8)
ax.plot([x0_ba, 1-x0_ba], [y0_ba, y0_ba], 'ro', markersize=8)

ax.set_xlim(-x_view, x_view)
ax.set_ylim(0, max(y0_ab, y0_ba, 1.1) * 1.3)
ax.set_title(f"Fixed Sum $\\alpha + \\beta={S}$, $\\alpha={alpha}$, $\\beta={beta}$ : Comparison of Order", fontsize=14)
ax.legend(loc="upper right")
ax.grid(True, linestyle=':', alpha=0.6)

st.pyplot(fig)

# --- 3. 解析情報の表示 ---
st.write(f"### Analysis for $\\alpha+\\beta = {S}$")
st.write(r"The $x$-value that gives a local maximum of $M_{\alpha} \circ M_{\beta}$ for $x > 1$:")
st.write( f"$x={x0_ab:.2f}, y={y0_ab:.4f}$")
st.write("---")
st.write(r"The $x$-value that gives a local maximum of $M_{\beta} \circ M_{\alpha}$ for $x > 1$:")
st.write( f"$x={x0_ba:.2f}, y={y0_ba:.4f}$")
st.write("---")

st.write("---")
st.write(f"$\\alpha + \\beta = {S}$")
st.write("---")
st.write(r"For $0 \leq \alpha< 1, 0< \beta < 1\ \text{and}\ \alpha+\beta<1$, we have")
st.latex(r"(M_{\alpha}\circ M_{\beta}) \left[ \chi_{[0,1]}\right](x) = \begin{cases} 1 & (0 \le x \le 1) \\ x^{\alpha-1} \left( 1+\frac{x^{\beta}-1}{\beta}\right) & (x > 1) \\ (1-x)^{\alpha-1} \left( 1+\frac{(1-x)^{\beta}-1}{\beta}\right) & (x < 0) \end{cases}")
st.write("---")
st.write(r"For $0\leq \alpha< 1$ and $\beta=0$, we have")
st.latex(r"(M_{\alpha}\circ M) \left[ \chi_{[0,1]}\right](x) = \begin{cases} 1 & (0 \le x \le 1) \\ x^{\alpha-1} \left( 1+\log(x)\right) & (x > 1) \\ (1-x)^{\alpha-1} \left( 1+\log(1-x)\right) & (x < 0) \end{cases}")
st.write("---")
if beta == 0:
    st.write(r"The $x$-value that gives a local maximum: $x_0 = e^{\frac{\alpha}{1-\alpha}}$ and $x_1 = 1-e^{\frac{\alpha}{1-\alpha}}$")
else:
    st.write(r"The $x$-value that gives a local maximum:")
    st.write(r"$x_0 = \left( \frac{(1-\alpha)(1-\beta)}{1-(\alpha+\beta)} \right)^{1/\beta}$ and $x_1 = 1-\left( \frac{(1-\alpha)(1-\beta)}{1-(\alpha+\beta)} \right)^{1/\beta}$.")




# 2. 現在のグラフ(fig)をPDF形式でバッファに保存
fig.savefig(buf, format="pdf", bbox_inches="tight")

# 3. ダウンロードボタンを設置
st.download_button(
    label="Export a graph to PDF",
    data=buf.getvalue(),
    file_name=f"Compositions of maximal operators behaviour:M_{alpha} and M_{beta}.pdf",
    mime="application/pdf"
)
