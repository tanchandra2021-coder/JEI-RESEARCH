print("9. FIGURES (JEI-Compliant)")
print("=" * 80)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'legend.framealpha': 1.0,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

C_BLUE   = '#2171B5'
C_ORANGE = '#D94801'
C_TEAL   = '#1B9E77'
C_PURPLE = '#7570B3'

MODEL_CLR = [C_BLUE, C_ORANGE, C_TEAL, C_PURPLE]

model_names_fig = ['Base\nFlash', 'Premium\nPro', 'Flash-\nLite', 'Fine-\ntuned']
x = np.arange(len(MODELS))
width = 0.35


# Figure 1 — Empathy Scores by Model and SES Level
fig, ax = plt.subplots(figsize=(8, 5.5))

high_means = [agg[agg['ses_level']=='high'][f'{m}_empathy'].mean() for m in MODELS]
low_means  = [agg[agg['ses_level']=='low'][f'{m}_empathy'].mean() for m in MODELS]
high_sds   = [agg[agg['ses_level']=='high'][f'{m}_empathy'].std() for m in MODELS]
low_sds    = [agg[agg['ses_level']=='low'][f'{m}_empathy'].std() for m in MODELS]

ax.bar(x - width/2, high_means, width, yerr=high_sds,
       label='High-SES', color=C_BLUE, capsize=5,
       edgecolor='white', linewidth=0.8,
       error_kw={'linewidth': 1.5, 'capthick': 1.5})
ax.bar(x + width/2, low_means, width, yerr=low_sds,
       label='Low-SES', color=C_ORANGE, capsize=5,
       edgecolor='white', linewidth=0.8,
       error_kw={'linewidth': 1.5, 'capthick': 1.5})

ax.set_ylabel('Mean Empathy Score')
ax.set_xticks(x)
ax.set_xticklabels(model_names_fig, fontweight='bold')
ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='upper right')
ax.set_ylim(0, max(max(high_means), max(low_means)) * 1.7)
ax.yaxis.grid(True, alpha=0.25, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)

for i in range(len(MODELS)):
    top = max(high_means[i] + high_sds[i], low_means[i] + low_sds[i])
    ax.annotate('ns', xy=(x[i], top + 0.03),
                ha='center', va='bottom', fontsize=14,
                fontstyle='italic', color='#555555')

plt.tight_layout()
plt.savefig('fig1_empathy_n48.png', dpi=300, bbox_inches='tight')
files.download('fig1_empathy_n48.png')
plt.show()
print("  Figure 1 saved.")


# Figure 2 — Response Equity: Semantic Similarity Between SES Pairs
fig, ax = plt.subplots(figsize=(8, 5.5))

sim_vals = [agg[f'{m}_semantic_sim'].mean() for m in MODELS]
sim_sds  = [agg[f'{m}_semantic_sim'].std() for m in MODELS]

bars = ax.bar(model_names_fig, sim_vals, yerr=sim_sds,
              color=MODEL_CLR, capsize=5,
              edgecolor='white', linewidth=0.8,
              error_kw={'linewidth': 1.5, 'capthick': 1.5})

ax.set_ylabel('Mean Semantic Similarity')
plt.xticks(fontweight='bold')
ax.set_ylim(0, 0.75)
ax.yaxis.grid(True, alpha=0.25, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)

for bar, val in zip(bars, sim_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('fig2_semantic_sim_n48.png', dpi=300, bbox_inches='tight')
files.download('fig2_semantic_sim_n48.png')
plt.show()
print("  Figure 2 saved.")


# Figure 3 — SES Bias Effect Sizes (Cohen's d)
fig, ax = plt.subplots(figsize=(8, 5))

effect_d = []
for m in MODELS:
    high = agg[agg['ses_level']=='high'][f'{m}_empathy']
    low  = agg[agg['ses_level']=='low'][f'{m}_empathy']
    pooled_std = np.sqrt(((len(high)-1)*high.var() + (len(low)-1)*low.var()) /
                         (len(high)+len(low)-2))
    d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0
    effect_d.append(d)

y_pos = np.arange(len(MODELS))

ax.axvspan(-0.2, 0.2, alpha=0.12, color=C_BLUE,
           label='Negligible effect zone (|d| < 0.2)', zorder=0)
ax.axvline(x=0, color='#333333', linestyle='-', linewidth=0.8, zorder=1)

bars = ax.barh(y_pos, effect_d, color=MODEL_CLR, height=0.6,
               edgecolor='white', linewidth=0.8, zorder=2)

ax.set_yticks(y_pos)
ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontweight='bold')
ax.set_xlabel("Cohen's d (SES Bias Effect Size)")
ax.legend(loc='lower left', frameon=True, fancybox=False,
          edgecolor='#CCCCCC', fontsize=11)
ax.xaxis.grid(True, alpha=0.25, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)

for bar, val in zip(bars, effect_d):
    offset = -0.015 if val < 0 else 0.015
    ha_val = 'right' if val < 0 else 'left'
    ax.text(val + offset, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha=ha_val, va='center',
            fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('fig3_ses_bias_n48.png', dpi=300, bbox_inches='tight')
files.download('fig3_ses_bias_n48.png')
plt.show()
print("  Figure 3 saved.")


# Figure 4 — Power Analysis
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(effect_sizes, power_values, color=C_BLUE, linewidth=2.5,
        label='Power curve', zorder=3)

ax.axhline(y=0.80, color='#888888', linestyle='--', linewidth=1.5,
           label='80% power threshold', zorder=1)
ax.axvline(x=d_for_80, color='#888888', linestyle=':', linewidth=1,
           alpha=0.5, zorder=1)

ax.fill_between(effect_sizes, power_values, 0.80,
                where=np.array(power_values) >= 0.80,
                alpha=0.12, color=C_BLUE, zorder=2)

markers = ['o', 's', 'D', '^']
for i, (name, d) in enumerate(observed.items()):
    p_at_d = power_analysis.solve_power(effect_size=d, nobs1=n_per_group,
                                         alpha=alpha, ratio=1.0)
    ax.plot(d, p_at_d, marker=markers[i], color=MODEL_CLR[i],
            markersize=10, markeredgecolor='white', markeredgewidth=1.5,
            zorder=4, label=f'{name} (d={d:.2f})')

ax.set_xlabel("Effect Size (Cohen's d)")
ax.set_ylabel('Statistical Power')
ax.legend(loc='center right', frameon=True, fancybox=False,
          edgecolor='#CCCCCC', fontsize=11)
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.05)
ax.yaxis.grid(True, alpha=0.25, linestyle='-', color='#CCCCCC')
ax.xaxis.grid(True, alpha=0.25, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)

ax.annotate(f'd = {d_for_80:.2f}', xy=(d_for_80, 0.80),
            xytext=(d_for_80 + 0.15, 0.72),
            fontsize=13, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
            color='#555555')

plt.tight_layout()
plt.savefig('fig4_power_n48.png', dpi=300, bbox_inches='tight')
files.download('fig4_power_n48.png')
plt.show()
print("  Figure 4 saved.")


# Figure 5 — Paneled: (A) Category Comparison + (B) Effect Size Variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]

cat_a_sims = [agg[agg['category']=='A'][f'{m}_semantic_sim'].mean() for m in MODELS]
cat_b_sims = [agg[agg['category']=='B'][f'{m}_semantic_sim'].mean() for m in MODELS]
cat_a_sds  = [agg[agg['category']=='A'][f'{m}_semantic_sim'].std() for m in MODELS]
cat_b_sds  = [agg[agg['category']=='B'][f'{m}_semantic_sim'].std() for m in MODELS]

ax.bar(x - width/2, cat_a_sims, width, yerr=cat_a_sds,
       label='Category A (Crisis)', color=C_ORANGE, capsize=5,
       edgecolor='white', linewidth=0.8,
       error_kw={'linewidth': 1.5, 'capthick': 1.5})
ax.bar(x + width/2, cat_b_sims, width, yerr=cat_b_sds,
       label='Category B (Advice)', color=C_BLUE, capsize=5,
       edgecolor='white', linewidth=0.8,
       error_kw={'linewidth': 1.5, 'capthick': 1.5})

ax.set_ylabel('Mean Semantic Similarity')
ax.set_xticks(x)
ax.set_xticklabels(model_names_fig, fontweight='bold')
ax.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC',
          loc='upper right', fontsize=11)
ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, alpha=0.25, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)

for i in range(len(MODELS)):
    top = max(cat_a_sims[i] + cat_a_sds[i], cat_b_sims[i] + cat_b_sds[i])
    ax.annotate('***', xy=(x[i], top + 0.02),
                ha='center', va='bottom', fontsize=14,
                fontweight='bold', color='#333333')

ax.text(-0.08, 1.05, 'A', transform=ax.transAxes,
        fontsize=20, fontweight='bold', va='top')

ax = axes[1]

run_ds = {MODEL_LABELS[m]: [] for m in MODELS}
for run_idx in range(6):
    run_file = f'gemini_4model_complete_results ({10 + run_idx}).csv'
    try:
        run_df = pd.read_csv(run_file)
    except FileNotFoundError:
        continue

    for m in MODELS:
        if f'{m}_empathy' in run_df.columns:
            high = run_df[run_df['ses_level']=='high'][f'{m}_empathy']
            low  = run_df[run_df['ses_level']=='low'][f'{m}_empathy']
            if len(high) > 1 and len(low) > 1:
                pooled = np.sqrt(((len(high)-1)*high.var() + (len(low)-1)*low.var()) /
                                 (len(high)+len(low)-2))
                d = (high.mean() - low.mean()) / pooled if pooled > 0 else 0
                run_ds[MODEL_LABELS[m]].append(d)

has_run_data = any(len(v) > 1 for v in run_ds.values())

if has_run_data:
    bp_data = [run_ds[MODEL_LABELS[m]] for m in MODELS]
    bp = ax.boxplot(bp_data, labels=[MODEL_LABELS[m] for m in MODELS],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='#333333', linewidth=2),
                    whiskerprops=dict(color='#666666', linewidth=1.2),
                    capprops=dict(color='#666666', linewidth=1.2),
                    flierprops=dict(marker='o', markerfacecolor='#999999',
                                   markersize=6, alpha=0.7))
    for patch, color in zip(bp['boxes'], MODEL_CLR):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#333333')
else:
    ax.bar(range(len(MODELS)), effect_d, color=MODEL_CLR,
           edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])

ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8)
ax.axhspan(-0.2, 0.2, alpha=0.10, color=C_BLUE)
ax.set_ylabel("Cohen's d (per run)")
ax.yaxis.grid(True, alpha=0.25, linestyle='-', color='#CCCCCC')
ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), fontweight='bold')

ax.text(-0.08, 1.05, 'B', transform=ax.transAxes,
        fontsize=20, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('fig5_category_and_variance.png', dpi=300, bbox_inches='tight')
files.download('fig5_category_and_variance.png')
plt.show()
print("  Figure 5 saved.")


print()
print("=" * 80)
print("ALL 5 FIGURES SAVED AND DOWNLOADED")
print("=" * 80)
print()
print("Files to submit to JEI:")
print("  fig1_empathy_n48.png")
print("  fig2_semantic_sim_n48.png")
print("  fig3_ses_bias_n48.png")
print("  fig4_power_n48.png")
print("  fig5_category_and_variance.png")
print()
print("Convert to .TIFF or .JPEG before submission if required.")
