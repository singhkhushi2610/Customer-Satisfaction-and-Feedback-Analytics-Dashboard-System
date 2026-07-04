"""
Customer Feedback Analytics System - Static HTML Dashboard Generator
Produces a single self-contained HTML file (no server, no admin rights needed).
Run with: python generate_dashboard.py
Then just double-click dashboard.html to open it in your browser.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

PRIMARY = "#2E5266"
ACCENT_POS = "#3E8E7E"
ACCENT_NEG = "#C1666B"
ACCENT_NEU = "#A9A9A9"

# ---------------- Load data ----------------
df = pd.read_csv("data/feedback_final_with_clusters.csv")
cat_summary = pd.read_csv("data/csi_by_category.csv")
tier_summary = pd.read_csv("data/csi_by_price_tier.csv")
cluster_profile = pd.read_csv("data/cluster_profile.csv")
top_pos = pd.read_csv("data/top_positive_keywords.csv")
top_neg = pd.read_csv("data/top_negative_keywords.csv")
sentiment_by_cat = pd.read_csv("data/sentiment_by_category.csv")

CLUSTER_NAMES = {
    0: "Core Performers",
    1: "At-Risk / Discount-Driven",
    2: "Premium Flagship",
    3: "Mainstream Budget",
}
df["segment"] = df["cluster"].map(CLUSTER_NAMES)
cluster_profile["segment"] = cluster_profile["cluster"].map(CLUSTER_NAMES)

# ---------------- KPIs ----------------
overall_csi = df["csi_score"].mean()
weighted_csi = np.average(df["csi_score"], weights=df["rating_count"])
avg_rating = df["rating"].mean()
pct_positive = (df["sentiment_label"] == "Positive").mean() * 100
n_products = len(df)

# ---------------- Charts ----------------
figs = {}

# 1. Satisfaction band distribution
band_counts = df["satisfaction_band"].value_counts().reset_index()
band_counts.columns = ["Band", "Count"]
band_order = ["Highly Satisfied", "Satisfied", "Neutral", "Dissatisfied"]
band_colors = {"Highly Satisfied": ACCENT_POS, "Satisfied": PRIMARY, "Neutral": ACCENT_NEU, "Dissatisfied": ACCENT_NEG}
figs["band"] = px.bar(band_counts, x="Band", y="Count", color="Band",
                       category_orders={"Band": band_order}, color_discrete_map=band_colors,
                       title="Satisfaction Band Distribution")
figs["band"].update_layout(showlegend=False, height=380)

# 2. CSI by price tier
figs["tier"] = px.bar(tier_summary, x="price_tier", y="csi_score", color="csi_score",
                       color_continuous_scale="Teal", title="CSI by Price Tier",
                       labels={"price_tier": "Price Tier", "csi_score": "CSI Score"})
figs["tier"].update_layout(height=380, coloraxis_showscale=False)

# 3. Rating distribution
figs["rating_dist"] = px.histogram(df, x="rating", nbins=20, color_discrete_sequence=[PRIMARY],
                                     title="Rating Distribution")
figs["rating_dist"].update_layout(height=350)

# 4. CSI by category
cat_sorted = cat_summary.sort_values("csi_score", ascending=True)
figs["cat_csi"] = px.bar(cat_sorted, x="csi_score", y="main_category", orientation="h",
                          color="csi_score", color_continuous_scale="Teal",
                          title="CSI Score by Category",
                          labels={"csi_score": "CSI Score", "main_category": "Category"},
                          hover_data=["n_products", "total_reviews"])
figs["cat_csi"].update_layout(height=480, coloraxis_showscale=False)

# 5. Sentiment mix by category
sbc_melted = sentiment_by_cat.melt(id_vars=sentiment_by_cat.columns[0], var_name="Sentiment", value_name="Percentage")
sbc_melted.columns = ["Category", "Sentiment", "Percentage"]
figs["sent_by_cat"] = px.bar(sbc_melted, x="Category", y="Percentage", color="Sentiment",
                              color_discrete_map={"Positive": ACCENT_POS, "Neutral": ACCENT_NEU, "Negative": ACCENT_NEG},
                              barmode="stack", title="Sentiment Mix by Category")
figs["sent_by_cat"].update_layout(height=420, xaxis_tickangle=-45)

# 6. Overall sentiment split
sent_counts = df["sentiment_label"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]
figs["sent_pie"] = px.pie(sent_counts, names="Sentiment", values="Count", hole=0.45,
                           color="Sentiment",
                           color_discrete_map={"Positive": ACCENT_POS, "Neutral": ACCENT_NEU, "Negative": ACCENT_NEG},
                           title="Overall Sentiment Split")
figs["sent_pie"].update_layout(height=380)

# 7. Sentiment vs rating scatter
sample_df = df.sample(min(500, len(df)), random_state=1)
figs["sent_scatter"] = px.scatter(sample_df, x="rating", y="vader_compound", opacity=0.4,
                                    color_discrete_sequence=[PRIMARY],
                                    title="Sentiment vs. Star Rating",
                                    labels={"rating": "Star Rating", "vader_compound": "VADER Sentiment Score"})
figs["sent_scatter"].update_layout(height=380)

# 8. Top positive keywords
figs["kw_pos"] = px.bar(top_pos.head(12).sort_values("tfidf_score"), x="tfidf_score", y="keyword",
                         orientation="h", color_discrete_sequence=[ACCENT_POS],
                         title="Top Keywords - Positive Reviews")
figs["kw_pos"].update_layout(height=420, xaxis_title="TF-IDF Score", yaxis_title="")

# 9. Top negative keywords
figs["kw_neg"] = px.bar(top_neg.head(12).sort_values("tfidf_score"), x="tfidf_score", y="keyword",
                         orientation="h", color_discrete_sequence=[ACCENT_NEG],
                         title="Top Keywords - Negative Reviews")
figs["kw_neg"].update_layout(height=420, xaxis_title="TF-IDF Score", yaxis_title="")

# 10. Segmentation scatter
figs["segments"] = px.scatter(df, x="discounted_price", y="rating", color="segment", log_x=True, opacity=0.5,
                               title="Customer Segments: Price vs. Rating",
                               labels={"discounted_price": "Discounted Price (Rs., log scale)", "rating": "Rating"},
                               color_discrete_sequence=px.colors.qualitative.Set2)
figs["segments"].update_layout(height=500)

# ---------------- Segment profile table ----------------
display_profile = cluster_profile[["segment", "n_products", "avg_price", "avg_rating", "avg_sentiment", "avg_discount"]].rename(columns={
    "segment": "Segment", "n_products": "# Products", "avg_price": "Avg Price (Rs.)",
    "avg_rating": "Avg Rating", "avg_sentiment": "Avg Sentiment", "avg_discount": "Avg Discount %"
})
profile_html_table = display_profile.to_html(index=False, classes="profile-table", border=0)

at_risk = cluster_profile[cluster_profile["segment"] == "At-Risk / Discount-Driven"]
at_risk_n = int(at_risk["n_products"].values[0]) if len(at_risk) else 0
at_risk_pct = at_risk_n / len(df) * 100 if len(df) else 0

# ---------------- Build HTML ----------------
def fig_div(fig, div_id):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)

html_parts = []
html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Customer Feedback Analytics System - Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #FAFBFC; margin: 0; padding: 0; color: #222; }}
    header {{ background: {PRIMARY}; color: white; padding: 24px 40px; }}
    header h1 {{ margin: 0; font-size: 26px; }}
    header p {{ margin: 4px 0 0 0; opacity: 0.85; font-size: 14px; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 40px; }}
    .kpi-row {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
    .kpi-card {{ flex: 1; min-width: 180px; background: white; border-left: 4px solid {PRIMARY};
                 border-radius: 6px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .kpi-card .label {{ font-size: 13px; color: #666; margin-bottom: 4px; }}
    .kpi-card .value {{ font-size: 26px; font-weight: 600; color: {PRIMARY}; }}
    .tabs {{ display: flex; gap: 4px; margin-bottom: 20px; border-bottom: 2px solid #E0E0E0; }}
    .tab-btn {{ padding: 10px 20px; cursor: pointer; background: none; border: none; font-size: 15px;
                color: #666; border-bottom: 3px solid transparent; }}
    .tab-btn.active {{ color: {PRIMARY}; border-bottom: 3px solid {PRIMARY}; font-weight: 600; }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .chart-row {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
    .chart-box {{ flex: 1; min-width: 400px; background: white; border-radius: 8px; padding: 12px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .chart-box.full {{ flex-basis: 100%; }}
    .insight-box {{ background: #EAF4F2; border-left: 4px solid {ACCENT_POS}; padding: 14px 18px;
                    border-radius: 6px; margin-top: 12px; font-size: 14px; }}
    .warning-box {{ background: #FBEAEA; border-left: 4px solid {ACCENT_NEG}; padding: 14px 18px;
                    border-radius: 6px; margin-top: 12px; font-size: 14px; }}
    table.profile-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    table.profile-table th, table.profile-table td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
    table.profile-table th {{ background: {PRIMARY}; color: white; }}
    footer {{ text-align: center; padding: 20px; color: #888; font-size: 12px; }}
</style>
</head>
<body>

<header>
    <h1>Customer Feedback Analytics System</h1>
    <p>Satisfaction, sentiment, and segmentation analysis for customer feedback data</p>
</header>

<div class="container">

<div class="kpi-row">
    <div class="kpi-card"><div class="label">Overall CSI</div><div class="value">{overall_csi:.1f} / 100</div></div>
    <div class="kpi-card"><div class="label">Weighted CSI</div><div class="value">{weighted_csi:.1f} / 100</div></div>
    <div class="kpi-card"><div class="label">Avg. Rating</div><div class="value">{avg_rating:.2f} ★</div></div>
    <div class="kpi-card"><div class="label">% Positive Sentiment</div><div class="value">{pct_positive:.1f}%</div></div>
    <div class="kpi-card"><div class="label">Products Analyzed</div><div class="value">{n_products:,}</div></div>
</div>

<div class="tabs">
    <button class="tab-btn active" onclick="showTab(event, 'overview')">Overview</button>
    <button class="tab-btn" onclick="showTab(event, 'category')">Category Analysis</button>
    <button class="tab-btn" onclick="showTab(event, 'sentiment')">Sentiment & Keywords</button>
    <button class="tab-btn" onclick="showTab(event, 'segmentation')">Segmentation</button>
</div>

<div id="overview" class="tab-content active">
    <div class="chart-row">
        <div class="chart-box">{fig_div(figs['band'], 'band')}</div>
        <div class="chart-box">{fig_div(figs['tier'], 'tier')}</div>
    </div>
    <div class="chart-row">
        <div class="chart-box full">{fig_div(figs['rating_dist'], 'rating_dist')}</div>
    </div>
</div>

<div id="category" class="tab-content">
    <div class="chart-row">
        <div class="chart-box full">{fig_div(figs['cat_csi'], 'cat_csi')}</div>
    </div>
    <div class="chart-row">
        <div class="chart-box full">{fig_div(figs['sent_by_cat'], 'sent_by_cat')}</div>
    </div>
</div>

<div id="sentiment" class="tab-content">
    <div class="chart-row">
        <div class="chart-box">{fig_div(figs['sent_pie'], 'sent_pie')}</div>
        <div class="chart-box">{fig_div(figs['sent_scatter'], 'sent_scatter')}</div>
    </div>
    <div class="chart-row">
        <div class="chart-box">{fig_div(figs['kw_pos'], 'kw_pos')}</div>
        <div class="chart-box">{fig_div(figs['kw_neg'], 'kw_neg')}</div>
    </div>
    <div class="insight-box">
        <strong>Insight:</strong> Negative reviews center on product reliability ("working", "battery")
        and service issues, while positive reviews emphasize quality, ease of use, and value for money.
    </div>
</div>

<div id="segmentation" class="tab-content">
    <div class="chart-row">
        <div class="chart-box full">{fig_div(figs['segments'], 'segments')}</div>
    </div>
    <h3>Segment Profiles</h3>
    {profile_html_table}
    <div class="warning-box">
        <strong>Retention priority:</strong> The 'At-Risk / Discount-Driven' segment ({at_risk_n} products,
        {at_risk_pct:.1f}% of catalog) shows negative average sentiment combined with the steepest
        average discounts - a strong candidate for quality review rather than further discounting.
    </div>
</div>

</div>

<footer>Customer Feedback Analytics System | MCA Major Project (23ONMCR-753) | Data: Amazon India Sales &amp; Customer Reviews Dataset, Kaggle</footer>

<script>
function showTab(evt, tabName) {{
    var contents = document.getElementsByClassName("tab-content");
    for (var i = 0; i < contents.length; i++) {{ contents[i].classList.remove("active"); }}
    var btns = document.getElementsByClassName("tab-btn");
    for (var i = 0; i < btns.length; i++) {{ btns[i].classList.remove("active"); }}
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
    window.dispatchEvent(new Event('resize'));
}}
</script>

</body>
</html>
""")

with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_parts[0])

print("Dashboard generated successfully -> dashboard.html")
print("Double-click dashboard.html to open it in your browser. No server needed.")