"""
================================================================================
THESIS ANALYSIS — COMPLETE SCRIPT
Market Power and Innovation in Spanish Manufacturing Firms:
Does Distance to the Technological Frontier Matter?

Author    : Ana Srugies Garcia
Degree    : Double Degree in International Studies and Economics, UC3M
Supervisor: Adelheid Holl
Submitted : June 2026

Python    : 3.12.6
================================================================================

LANGUAGE CONVENTION (supervisor's instruction — applied throughout):
  - "Labour productivity"  = revenue per employee. A single-factor measure.
                             Abbreviated LP. NOT called TFP, NOT called
                             "productivity" without qualification.
  - "TFP" (Total Factor    = productive efficiency estimated via the
    Productivity)            Levinsohn-Petrin (2003) method, controlling for
                             both labour and capital. A multi-factor measure.
                             NOT the same as labour productivity.
  These two measures are the two dependent variables of the thesis.
  They are related but methodologically distinct and must never be conflated.

SCRIPT STRUCTURE:
  STEP 1  — Configuration: file paths and global parameters
  STEP 2  — Load and stack the 6 ORBIS Excel batches
  STEP 3  — Reshape from wide to long format
  STEP 4  — Data cleaning
  STEP 5  — Construct labour productivity and its growth rate
  STEP 6  — Construct market power measures (PCM, OPM)
  STEP 7  — Construct distance to the labour productivity frontier
  STEP 8  — Estimate TFP via Levinsohn-Petrin (2003), sector by sector
  STEP 9  — Construct TFP growth and distance to the TFP frontier
  STEP 10 — Construct control variables
  STEP 11 — Lag all independent variables by one period
  STEP 12 — Descriptive statistics
  STEP 13 — Regression helper functions
  STEP 14 — Run all regression specifications
  STEP 15 — Marginal effects of PCM at different TFP-frontier distances
  STEP 16 — Print formatted regression tables
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import warnings
from itertools import product as cartesian_product

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 130)


# ==============================================================================
'''----Step 1: Configuration----
'''
# ==============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
# >>> CHANGE THESE TO YOUR LOCAL PATHS <<<
BATCH_FOLDER  = r"C:\Users\anasr\Desktop\TFGS\TFG ECONOMIA 2\02. Data"

OUTPUT_FOLDER = r"C:\Users\anasr\Desktop\TFGS\TFG ECONOMIA 2\02. Data\01. Outputs"

# File name pattern — ORBIS exports are named Export_01.xlsx … Export_06.xlsx
BATCH_PATTERN = "Export_{:02d}.xlsx"
N_BATCHES     = 6

# ── Sample restrictions ───────────────────────────────────────────────────────
YEARS_KEEP         = list(range(2015, 2024))  # 2015–2023 (pre-2015 too sparse)
MIN_EMPLOYEES      = 10      # already filtered in ORBIS; verified here
MIN_CONSEC_YEARS   = 3       # minimum consecutive years of data per firm
MIN_SECTOR_FIRMS   = 10      # min firms per NACE-2 × year for a valid frontier

# ── LP estimation ─────────────────────────────────────────────────────────────
MIN_OBS_FOR_LP_EST = 100     # minimum observations per sector for LP estimation
LP_POLY_DEGREE     = 3       # degree of polynomial in (ln K, ln M) — Stage 1

# ── Winsorisation ─────────────────────────────────────────────────────────────
WINS_LO = 0.01               # 1st percentile
WINS_HI = 0.99               # 99th percentile

# ── NACE Rev. 2 — 2-digit sector labels ───────────────────────────────────────
NACE2_LABELS = {
    10: 'Food products',         11: 'Beverages',
    12: 'Tobacco products',      13: 'Textiles',
    14: 'Wearing apparel',       15: 'Leather products',
    16: 'Wood products',         17: 'Paper',
    18: 'Printing',              19: 'Petroleum products',
    20: 'Chemicals',             21: 'Pharmaceuticals',
    22: 'Rubber & plastics',     23: 'Non-metallic minerals',
    24: 'Basic metals',          25: 'Fabricated metals',
    26: 'Electronics',           27: 'Electrical equipment',
    28: 'Machinery & equipment', 29: 'Motor vehicles',
    30: 'Other transport',       31: 'Furniture',
    32: 'Other manufacturing',   33: 'Repair & installation',
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("=" * 70)
print("THESIS ANALYSIS — Spanish Manufacturing Firms, 2015–2023")
print("=" * 70)
print(f"  Batch folder : {BATCH_FOLDER}")
print(f"  Output folder: {OUTPUT_FOLDER}")


# ==============================================================================
'''----Step 2: Load and stack the 6 ORBIS Excel batches----
ORBIS exports the 15,719 firms split across 6 Excel files (export limit is
2,739 firms per file). Each file has two sheets: "Search summary" (metadata)
and "Results" (the actual data). We read only the "Results" sheet.

The first column of Results is a row-number index added by ORBIS — we drop it.
Pandas auto-renames duplicate column names with a ".1" suffix (ORBIS includes
some variables twice under different sections). We keep only the first
occurrence of each duplicate by dropping every column whose name ends in ".1".

All six DataFrames are concatenated into one wide DataFrame called `raw`.
'''
# ==============================================================================

print("\n[Step 2] Loading ORBIS batches...")

batches = []
for i in range(1, N_BATCHES + 1):
    fpath = os.path.join(BATCH_FOLDER, BATCH_PATTERN.format(i))
    batch = pd.read_excel(
        fpath,
        engine='calamine',     # fast engine for xlsx; pip install python-calamine
        sheet_name='Results',
        header=0,
    )
    batch = batch.iloc[:, 1:]                                  # drop ORBIS index col
    batch = batch[[c for c in batch.columns                    # drop .1 duplicates
                   if not str(c).endswith('.1')]]
    batches.append(batch)
    print(f"  Batch {i}: {len(batch):,} firms")

raw = pd.concat(batches, ignore_index=True)
print(f"  Combined: {len(raw):,} firms  |  {raw.shape[1]} columns")


# ==============================================================================
'''----Step 3: Reshape from wide to long format----
ORBIS delivers one row per firm with separate columns for each year
(e.g. "Operating revenue (Turnover) EUR 2023", "… EUR 2022", …).
We reshape this into a panel: one row per firm-year combination.

Static columns (firm identifiers, legal form, etc.) are repeated for every year.
Financial columns are identified by matching their prefix string to the year
suffix. The full year range in the download is 2008–2023; we carry all years
through the reshape and filter to 2015–2023 in Step 4.
'''
# ==============================================================================

print("\n[Step 3] Reshaping wide → long...")

# ── Static (firm-level) columns ───────────────────────────────────────────────
STATIC_COLS = [
    'Company name Latin alphabet',
    'BvD ID number',           # main firm identifier throughout
    'National ID',
    'Country ISO code',
    'NACE Rev. 2, core code (4 digits)',
    'Consolidation code',
    'Last avail. year',
    'Status',
    'Date of incorporation',
    'Standardized legal form',
]

# ── Time-varying variable stems ───────────────────────────────────────────────
# Format in ORBIS: "<Stem> <Year>", e.g. "Operating revenue (Turnover) EUR 2023"
FIN_VARS = {
    # short name            : ORBIS column prefix
    'closing_date'          : 'Closing date',
    'num_months'            : 'Number of months',
    'revenue'               : 'Operating revenue (Turnover) EUR',
    'added_value'           : 'Added value EUR',
    'ebitda'                : 'EBITDA EUR',
    'material_costs'        : 'Material costs EUR',
    'staff_costs'           : 'Costs of employees EUR',
    'da'                    : 'Depreciation & Amortization EUR',
    'export_rev'            : 'Export revenue EUR',
    'intangibles'           : 'Intangible assets EUR',
    'tangibles'             : 'Tangible fixed assets EUR',
    'total_assets'          : 'Total assets EUR',
    'equity'                : 'Shareholders funds EUR',
    'lt_debt'               : 'Long term debt EUR',
    'st_debt'               : 'Loans & short-term debt EUR',
    'employees'             : 'Number of employees',
    'rd_expenses'           : 'Research & Development expenses EUR',
}

ALL_YEARS = list(range(2008, 2024))   # full ORBIS range; filtered later

rows = []
for _, firm in raw.iterrows():
    base = {col: firm[col] for col in STATIC_COLS if col in firm.index}
    for yr in ALL_YEARS:
        row = base.copy()
        row['year'] = yr
        for short_name, prefix in FIN_VARS.items():
            col_name = f"{prefix} {yr}"
            row[short_name] = firm[col_name] if col_name in firm.index else np.nan
        rows.append(row)

long = pd.DataFrame(rows)
print(f"  Long format: {len(long):,} obs  |  {long.shape[1]} columns")

# Convert all financial columns to numeric (ORBIS may store as strings)
for col in FIN_VARS:
    long[col] = pd.to_numeric(long[col], errors='coerce')


# ==============================================================================
'''----Step 4: Data cleaning----
We apply the following cleaning steps in order:

  4a. Keep only annual accounts (num_months == 12). Accounts covering fewer
      or more than 12 months would make year-on-year comparisons invalid.

  4b. Restrict to 2015–2023. Pre-2015 observations cover fewer than 10% of
      firms (< 1,300 firms vs ~15,700 from 2015), making earlier years
      unreliable for panel estimation.

  4c. Drop firm-years with missing or non-positive revenue or employees.
      These are either shell companies or data errors.

  4d. Verify the 10-employee minimum: keep only firms that reached at least
      10 employees in at least one year (filter already applied in ORBIS but
      verified here for integrity).

  4e. Require at least 3 consecutive years of data per firm. This ensures
      that first-differencing and lagging (Steps 5, 11) produce usable obs.

  4f. Extract 2-digit NACE code and firm age from raw fields.
'''
# ==============================================================================

print("\n[Step 4] Cleaning...")

df = long.copy()

# 4a. Keep only annual accounts
n_before = len(df)
df = df[df['num_months'].isna() | (df['num_months'] == 12)]
print(f"  4a. Non-annual accounts removed : {n_before - len(df):>8,} obs")

# 4b. Restrict to 2015–2023
n_before = len(df)
df = df[df['year'].isin(YEARS_KEEP)]
print(f"  4b. Pre-2015 years removed       : {n_before - len(df):>8,} obs")

# 4c. Drop missing / non-positive revenue or employees
n_before = len(df)
df = df[(df['revenue'] > 0) & (df['employees'] > 0)]
print(f"  4c. Missing/zero revenue or emp  : {n_before - len(df):>8,} obs")

# 4d. Verify 10-employee minimum (at least one year)
firm_max_emp = df.groupby('BvD ID number')['employees'].max()
valid_firms  = firm_max_emp[firm_max_emp >= MIN_EMPLOYEES].index
n_before     = df['BvD ID number'].nunique()
df = df[df['BvD ID number'].isin(valid_firms)]
print(f"  4d. Firms never ≥ {MIN_EMPLOYEES} employees  : "
      f"{n_before - df['BvD ID number'].nunique():>8,} firms")

# 4e. Require at least MIN_CONSEC_YEARS consecutive years
df = df.sort_values(['BvD ID number', 'year'])

def _has_n_consecutive(year_series, n):
    """Return True if the sorted year list contains a run of length ≥ n."""
    yrs   = sorted(year_series)
    count = 1
    for i in range(1, len(yrs)):
        count = count + 1 if yrs[i] == yrs[i - 1] + 1 else 1
        if count >= n:
            return True
    return False

consec_ok   = df.groupby('BvD ID number')['year'].apply(
    lambda y: _has_n_consecutive(y, MIN_CONSEC_YEARS)
)
valid_consec = consec_ok[consec_ok].index
n_before     = df['BvD ID number'].nunique()
df = df[df['BvD ID number'].isin(valid_consec)]
print(f"  4e. < {MIN_CONSEC_YEARS} consecutive years     : "
      f"{n_before - df['BvD ID number'].nunique():>8,} firms")

# 4f. NACE codes and firm age
df['nace4']      = df['NACE Rev. 2, core code (4 digits)'].astype(str).str.zfill(4)
df['nace2']      = df['nace4'].str[:2].astype(int)
df['nace2_label'] = df['nace2'].map(NACE2_LABELS)

df['Date of incorporation'] = pd.to_datetime(
    df['Date of incorporation'], errors='coerce'
)
df['inc_year']  = df['Date of incorporation'].dt.year
df['firm_age']  = (df['year'] - df['inc_year']).clip(lower=0)

df = df.sort_values(['BvD ID number', 'year']).reset_index(drop=True)
print(f"\n  Clean panel: {df['BvD ID number'].nunique():,} firms  |  "
      f"{len(df):,} obs  |  {df['year'].min()}–{df['year'].max()}")


# ==============================================================================
'''----Step 5: Labour productivity and its growth rate----
Labour productivity (LP) = revenue per employee.
  - Revenue is in EUR thousands (as downloaded from ORBIS).
  - Employees is a headcount.
  - LP is therefore in EUR thousands per employee.
  - We work with log(LP) throughout.

Labour productivity growth = first difference of log(LP) within firm:
  ΔlnLP_it = ln(LP_it) − ln(LP_it−1)

This is one of the two dependent variables of the thesis.
It is a SINGLE-FACTOR measure — it does not control for capital or
intermediate inputs and should not be confused with TFP.

We winsorise ΔlnLP within each year at the 1st and 99th percentiles
to remove extreme outliers driven by data errors or exceptional events.
'''
# ==============================================================================

print("\n[Step 5] Labour productivity and labour productivity growth...")

df['ln_lp'] = np.log(df['revenue'] / df['employees'])

df['dlnlp'] = df.groupby('BvD ID number')['ln_lp'].diff()

# Winsorise ΔlnLP within year
df['dlnlp'] = df.groupby('year')['dlnlp'].transform(
    lambda s: s.clip(s.quantile(WINS_LO), s.quantile(WINS_HI))
)

print(f"  ln(Labour productivity): {df['ln_lp'].notna().sum():,} valid obs")
print(f"  Labour productivity growth (ΔlnLP): "
      f"{df['dlnlp'].notna().sum():,} valid obs")
print(f"    mean = {df['dlnlp'].mean():.4f}  |  "
      f"sd = {df['dlnlp'].std():.4f}  |  "
      f"median = {df['dlnlp'].median():.4f}")


# ==============================================================================
'''----Step 6: Market power measures (PCM and OPM)----
We construct two alternative measures of firm-level market power.

PCM — Price-Cost Margin (primary measure):
  PCM_it = (Revenue_it − Material costs_it − Staff costs_it) / Revenue_it

  This is the standard Lerner-adjacent measure used in this literature
  (Aghion et al., 2005; Benavente & Zuñiga, 2022). It captures the share
  of revenue not absorbed by variable costs, approximating the mark-up
  over marginal cost.

OPM — Operating Profit Margin (robustness measure):
  EBIT_it = EBITDA_it − Depreciation & Amortisation_it
  OPM_it  = EBIT_it / Revenue_it

  OPM includes capital depreciation and is therefore a tighter margin
  concept. Used only in robustness specifications.

Both measures are:
  (1) Capped at economically impossible values (> 1 or < −5) before
      winsorisation.
  (2) Winsorised within year at 1st–99th percentiles.
  (3) Centred at their sample mean before use in regressions. Centring
      reduces the multicollinearity between the linear and squared terms
      without changing the estimated marginal effects.
'''
# ==============================================================================

print("\n[Step 6] Market power measures (PCM and OPM)...")

# PCM
df['pcm'] = (
    (df['revenue'] - df['material_costs'] - df['staff_costs'])
    / df['revenue']
)
df.loc[df['pcm'] >  1, 'pcm'] = np.nan    # economically impossible
df.loc[df['pcm'] < -5, 'pcm'] = np.nan

# OPM
df['ebit'] = df['ebitda'] - df['da']
df['opm']  = df['ebit'] / df['revenue']
df.loc[df['opm'] >  1, 'opm'] = np.nan
df.loc[df['opm'] < -5, 'opm'] = np.nan

# Winsorise within year
for var in ['pcm', 'opm']:
    df[var] = df.groupby('year')[var].transform(
        lambda s: s.clip(s.quantile(WINS_LO), s.quantile(WINS_HI))
    )

# Centre at sample mean
pcm_mean = df['pcm'].mean()
opm_mean = df['opm'].mean()
df['pcm_c'] = df['pcm'] - pcm_mean
df['opm_c'] = df['opm'] - opm_mean

print(f"  PCM : {df['pcm'].notna().sum():,} obs  |  "
      f"mean = {pcm_mean:.4f}  |  sd = {df['pcm'].std():.4f}")
print(f"  OPM : {df['opm'].notna().sum():,} obs  |  "
      f"mean = {opm_mean:.4f}  |  sd = {df['opm'].std():.4f}")
print(f"  PCM and OPM centred at their respective sample means.")


# ==============================================================================
'''----Step 7: Distance to the labour productivity frontier----
The labour productivity frontier is defined as the 95th percentile of
log(labour productivity) within each NACE-2 × year cell:
  LP_frontier_{j,t} = P95(ln_lp_{i,t})  for all firms i in sector j in year t

Distance to the LP frontier:
  dist_lp_{i,t} = LP_frontier_{j,t} − ln_lp_{i,t}

By construction, dist_lp ≥ 0. A value of 0 means the firm IS the frontier.
A higher value means the firm is further below the frontier in terms of
labour productivity.

This LP-based frontier and distance measure is used ONLY in the labour
productivity growth robustness specifications (Specs 1 and 2). The main
specifications use a TFP-based frontier (Step 9).

NACE-2 × year cells with fewer than MIN_SECTOR_FIRMS firms are dropped
because the P95 is unreliable in thin cells.
'''
# ==============================================================================

print("\n[Step 7] Distance to the labour productivity frontier...")

# Frontier: P95 of ln(LP) within NACE-2 × year
df['lp_frontier'] = df.groupby(['nace2', 'year'])['ln_lp'].transform(
    lambda x: x.quantile(0.95)
)

# Distance: non-negative by construction
df['dist_lp_frontier'] = df['lp_frontier'] - df['ln_lp']

# Drop thin cells (unreliable frontier)
cell_size = df.groupby(['nace2', 'year'])['BvD ID number'].transform('nunique')
n_before  = len(df)
df = df[cell_size >= MIN_SECTOR_FIRMS]
print(f"  Thin NACE-2×year cells removed: {n_before - len(df):,} obs")

print(f"  Distance to LP frontier: {df['dist_lp_frontier'].notna().sum():,} obs  |  "
      f"mean = {df['dist_lp_frontier'].mean():.4f}  |  "
      f"sd = {df['dist_lp_frontier'].std():.4f}")


# ==============================================================================
'''----Step 8: TFP estimation via Levinsohn-Petrin (2003), sector by sector----
We estimate Total Factor Productivity (TFP) using the Levinsohn-Petrin (2003)
method. TFP is a MULTI-FACTOR measure of productive efficiency — unlike labour
productivity, it controls for both labour (L) and capital (K) inputs.

Production function (value-added specification):
  ln(VA)_it = β_l · ln(L)_it + β_k · ln(K)_it + ω_it + ε_it

where:
  VA   = added value (EUR thousands) — output net of intermediate inputs
  L    = number of employees — labour input
  K    = tangible fixed assets (EUR thousands) — capital proxy
  ω_it = TFP — the firm-year-specific efficiency term we want to recover
  ε_it = i.i.d. measurement error

The identification problem: ω is observed by the firm (affects its input
choices) but not by the econometrician. OLS is therefore biased.

Levinsohn-Petrin solution: use intermediate inputs (material costs M) as a
proxy for ω. Under the assumption that M = m(ω, K) is strictly monotone in ω,
we can invert it to express ω as a function of (K, M).

Stage 1 — Consistent estimate of β_l:
  Regress ln(VA) on ln(L) and a 3rd-degree polynomial in (ln(K), ln(M))
  plus year dummies. The polynomial absorbs ω and β_k · K together.
  → Recovers β_l consistently (ln(L) is uncorrelated with ε once ω is
    controlled for by the polynomial).
  → Recovers φ_it = β_k · ln(K)_it + ω_it  (composite residual).

Stage 2 — Consistent estimate of β_k:
  For a candidate β_k, compute ω_it(β_k) = φ_it − β_k · ln(K)_it.
  TFP follows a first-order Markov process: ω_it = g(ω_it-1) + ξ_it.
  We approximate g(·) with a cubic polynomial in lagged ω.
  The residual ξ_it (innovation in TFP) should be uncorrelated with the
  firm's capital stock (which was chosen before ξ was realised):
    Moment condition: E[ξ_it · ln(K)_it] = 0.
  We find β_k that satisfies this moment condition using a scalar root
  search (Brent's method). If the moment does not change sign over the
  search interval, we fall back to OLS of φ on ln(K).

TFP:
  ω_it = ln(VA)_it − β_l · ln(L)_it − β_k · ln(K)_it

Estimation is done separately for each 2-digit NACE sector. Sectors with
fewer than MIN_OBS_FOR_LP_EST observations are skipped.

IMPORTANT: ω_it is TFP — it is NOT labour productivity. Do not conflate them.
'''
# ==============================================================================

print("\n[Step 8] Estimating TFP via Levinsohn-Petrin (2003)...")
print(f"  Polynomial degree: {LP_POLY_DEGREE}  |  "
      f"Min obs per sector: {MIN_OBS_FOR_LP_EST}")

# Build LP estimation subsample: all inputs must be positive
df_lp = df[
    (df['added_value']    > 0) &
    (df['material_costs'] > 0) &
    (df['tangibles']      > 0) &
    (df['employees']      > 0)
].copy()

# Log-transform inputs
df_lp['y'] = np.log(df_lp['added_value'])      # output: log value added
df_lp['l'] = np.log(df_lp['employees'])         # labour input
df_lp['k'] = np.log(df_lp['tangibles'])         # capital proxy
df_lp['m'] = np.log(df_lp['material_costs'])    # LP proxy (intermediate inputs)


def _poly_basis(sub_df, k_col, m_col, degree):
    """
    Build a polynomial basis in (k, m) up to total degree `degree`.
    E.g. degree=3 includes: k, m, k², km, m², k³, k²m, km², m³.
    Cross terms with p+q=0 (constants) are excluded — absorbed by intercept.
    """
    terms = {}
    for p, q in cartesian_product(range(degree + 1), range(degree + 1)):
        if 1 <= p + q <= degree:
            terms[f'k{p}m{q}'] = sub_df[k_col] ** p * sub_df[m_col] ** q
    return pd.DataFrame(terms, index=sub_df.index)


def _lp_estimate_one_sector(sub_df):
    """
    Run the full Levinsohn-Petrin procedure for a single sector.
    Returns a DataFrame with columns [BvD ID number, year, tfp, beta_l, beta_k]
    or None if there are too few observations.
    """
    sub = sub_df.dropna(subset=['y', 'l', 'k', 'm']).copy()
    if len(sub) < MIN_OBS_FOR_LP_EST:
        return None

    # ── Stage 1: OLS of y on l + poly(k, m) + year dummies ──────────────────
    year_dummies = pd.get_dummies(
        sub['year'], prefix='yr', drop_first=True
    ).astype(float)
    poly = _poly_basis(sub, 'k', 'm', LP_POLY_DEGREE)

    X1 = pd.concat([sub[['l']], poly, year_dummies], axis=1).astype(float)
    X1 = X1.loc[:, X1.std() > 1e-8]        # remove zero-variance cols
    X1.insert(0, 'const', 1.0)             # add intercept

    try:
        beta1, *_ = np.linalg.lstsq(X1.values, sub['y'].values, rcond=None)
    except Exception:
        return None

    beta_l    = beta1[list(X1.columns).index('l')]
    sub['phi'] = sub['y'] - beta_l * sub['l']   # φ = β_k·k + ω

    # ── Stage 2: GMM moment condition E[ξ · k] = 0 ───────────────────────────
    sub = sub.sort_values(['BvD ID number', 'year'])
    sub['lag_phi'] = sub.groupby('BvD ID number')['phi'].shift(1)
    sub['lag_k']   = sub.groupby('BvD ID number')['k'].shift(1)
    sub2 = sub.dropna(subset=['lag_phi', 'lag_k'])

    def _moment(bk):
        """Evaluate the GMM moment at candidate capital elasticity bk."""
        omega     = sub2['phi']     - bk * sub2['k']
        lag_omega = sub2['lag_phi'] - bk * sub2['lag_k']
        # Approximate the Markov transition g(ω_t-1) with a cubic poly
        lo    = lag_omega.values
        basis = np.column_stack([np.ones(len(lo)), lo, lo ** 2, lo ** 3])
        try:
            coef, *_ = np.linalg.lstsq(basis, omega.values, rcond=None)
            e_omega  = basis @ coef
        except Exception:
            e_omega  = np.full(len(omega), omega.mean())
        xi = omega.values - e_omega          # innovation in TFP
        return np.mean(xi * sub2['k'].values)

    # Root search over a plausible range of capital elasticities
    try:
        m_lo, m_hi = _moment(-1.5), _moment(1.5)
        if m_lo * m_hi < 0:
            beta_k = brentq(_moment, -1.5, 1.5, xtol=1e-4, maxiter=200)
        else:
            # Fallback: OLS of φ on k (consistent if ω ⊥ k in expectation)
            coef_fallback = np.polyfit(sub['k'].values, sub['phi'].values, 1)
            beta_k        = max(0.0, coef_fallback[0])
    except Exception:
        beta_k = 0.20    # neutral fallback near typical capital elasticity

    # ── Recover TFP ──────────────────────────────────────────────────────────
    sub['tfp']    = sub['y'] - beta_l * sub['l'] - beta_k * sub['k']
    sub['beta_l'] = beta_l
    sub['beta_k'] = beta_k

    return sub[['BvD ID number', 'year', 'tfp', 'beta_l', 'beta_k']]


# Run estimation sector by sector
tfp_parts    = []
sector_betas = []

print(f"\n  {'NACE':<6} {'Sector':<28} {'N obs':>7}  "
      f"{'β_labour':>9}  {'β_capital':>10}  {'RTS':>6}")
print("  " + "-" * 68)

for nace2_val, grp in df_lp.groupby('nace2'):
    result = _lp_estimate_one_sector(grp)
    label  = NACE2_LABELS.get(nace2_val, str(nace2_val))
    if result is not None:
        bl  = result['beta_l'].iloc[0]
        bk  = result['beta_k'].iloc[0]
        rts = bl + bk
        tfp_parts.append(result)
        sector_betas.append({'nace2': nace2_val, 'label': label,
                             'N_obs': len(result), 'beta_l': bl,
                             'beta_k': bk, 'RTS': rts})
        print(f"  {nace2_val:<6} {label[:27]:<28} {len(result):>7,}  "
              f"{bl:>+9.3f}  {bk:>+10.3f}  {rts:>6.3f}")
    else:
        print(f"  {nace2_val:<6} {label[:27]:<28} {'—':>7}  "
              f"  (skipped — insufficient obs)")

tfp_df      = pd.concat(tfp_parts, ignore_index=True)
betas_table = pd.DataFrame(sector_betas).set_index('nace2')

print(f"\n  TFP estimated: {tfp_df['BvD ID number'].nunique():,} firms  |  "
      f"{len(tfp_df):,} obs")
print(f"  Sectors estimated: {len(sector_betas)}")

# Merge TFP back onto main panel
df = df.merge(
    tfp_df[['BvD ID number', 'year', 'tfp']],
    on=['BvD ID number', 'year'],
    how='left',
)
df = df.sort_values(['BvD ID number', 'year']).reset_index(drop=True)
print(f"  TFP merged: {df['tfp'].notna().sum():,} obs with TFP")


# ==============================================================================
'''----Step 9: TFP growth and distance to the TFP frontier----
TFP growth = first difference of TFP within firm:
  ΔlnTFP_it = TFP_it − TFP_it−1
  (TFP is already in logs from Step 8, as ln(VA) − β_l·ln(L) − β_k·ln(K))

TFP frontier = P95 of TFP within each NACE-2 × year cell:
  TFP_frontier_{j,t} = P95(TFP_{i,t})  for all i in sector j, year t

Distance to the TFP frontier:
  dist_tfp_{i,t} = TFP_frontier_{j,t} − TFP_{i,t}

This is analogous to the LP-based frontier in Step 7 but uses TFP rather
than labour productivity as the productivity concept. Because TFP controls
for capital and labour, it is a theoretically cleaner measure of a firm's
position relative to the best-practice firm in its sector.

Both ΔlnTFP (dependent variable) and dist_tfp (moderating variable) in
the main specifications use this TFP-based concept.

ΔlnTFP is winsorised within year at 1st–99th percentiles.
'''
# ==============================================================================

print("\n[Step 9] TFP growth and distance to the TFP frontier...")

# TFP growth (first difference within firm)
df['dlntfp'] = df.groupby('BvD ID number')['tfp'].diff()

# Winsorise within year
df['dlntfp'] = df.groupby('year')['dlntfp'].transform(
    lambda s: s.clip(s.quantile(WINS_LO), s.quantile(WINS_HI))
)

# TFP frontier: P95 within NACE-2 × year
df['tfp_frontier']    = df.groupby(['nace2', 'year'])['tfp'].transform(
    lambda x: x.quantile(0.95)
)
df['dist_tfp_frontier'] = df['tfp_frontier'] - df['tfp']

print(f"  TFP growth (ΔlnTFP): {df['dlntfp'].notna().sum():,} obs  |  "
      f"mean = {df['dlntfp'].mean():.4f}  |  sd = {df['dlntfp'].std():.4f}")
print(f"  Correlation(ΔlnTFP, ΔlnLP): "
      f"{df[['dlntfp', 'dlnlp']].corr().iloc[0, 1]:.3f}  "
      f"(related but distinct measures)")
print(f"  Distance to TFP frontier: {df['dist_tfp_frontier'].notna().sum():,} obs  |  "
      f"mean = {df['dist_tfp_frontier'].mean():.4f}  |  "
      f"sd = {df['dist_tfp_frontier'].std():.4f}")


# ==============================================================================
'''----Step 10: Control variables----
The following firm-level controls are included in all specifications,
lagged one period (Step 11):

  log(Employees)    — firm size proxy. Larger firms may innovate more due
                       to scale economies (Scherer, 1965).
  Firm age          — years since incorporation. Older firms may have
                       accumulated knowledge but face inertia.
  Leverage          — (long-term debt + short-term debt) / total assets.
                       Financial constraints can limit investment in
                       productivity-enhancing activities.
  Capital intensity — log(tangible fixed assets / employees). Controls for
                       factor intensity differences across firms.

Note: Export revenue is entirely missing from this ORBIS extract (all NaN).
The export dummy planned in the original methodology is therefore omitted
from all specifications. This is acknowledged as a data limitation.
'''
# ==============================================================================

print("\n[Step 10] Control variables...")

df['log_employees']   = np.log(df['employees'])
df['leverage']        = (
    (df['lt_debt'] + df['st_debt']) / df['total_assets']
).clip(lower=0, upper=5)     # cap at 5 to avoid leverage outliers
df['capital_intensity'] = np.log(
    (df['tangibles'] / df['employees']).clip(lower=1)
)
# firm_age already constructed in Step 4f

n_missing_export = df['export_rev'].isna().sum()
print(f"  Export revenue missing: {n_missing_export:,} obs "
      f"({100*n_missing_export/len(df):.1f}%) — export dummy OMITTED")
print(f"  log(Employees)   : {df['log_employees'].notna().sum():,} obs  |  "
      f"mean = {df['log_employees'].mean():.3f}")
print(f"  Leverage         : {df['leverage'].notna().sum():,} obs  |  "
      f"mean = {df['leverage'].mean():.3f}")
print(f"  Capital intensity: {df['capital_intensity'].notna().sum():,} obs  |  "
      f"mean = {df['capital_intensity'].mean():.3f}")
print(f"  Firm age         : {df['firm_age'].notna().sum():,} obs  |  "
      f"mean = {df['firm_age'].mean():.1f} years")


# ==============================================================================
'''----Step 11: Lag all independent variables by one period----
All regressors — market power measures, distance-to-frontier variables,
and controls — are lagged by one year (t−1). This follows the standard
practice in the literature (Aghion et al., 2005; Benavente & Zuñiga, 2022).

Lagging serves two purposes:
  1. Reduces simultaneity bias (current innovation should not affect last
     year's market power or frontier position).
  2. Provides a natural time ordering consistent with the theory.

After lagging, we construct:
  - Squared terms of centred PCM and OPM (for the quadratic baseline specs)
  - Interaction terms: PCM × dist_tfp, PCM × dist_lp, OPM × dist_tfp
    (for the frontier heterogeneity main specs)

We also create the industry × year FE identifier used in regressions.
'''
# ==============================================================================

print("\n[Step 11] Lagging independent variables...")

VARS_TO_LAG = [
    'pcm',   'pcm_c',
    'opm',   'opm_c',
    'dist_lp_frontier',
    'dist_tfp_frontier',
    'log_employees',
    'firm_age',
    'leverage',
    'capital_intensity',
]

for var in VARS_TO_LAG:
    df[f'L_{var}'] = df.groupby('BvD ID number')[var].shift(1)

# Squared terms of centred lagged measures
df['L_pcm_c_sq'] = df['L_pcm_c'] ** 2
df['L_opm_c_sq'] = df['L_opm_c'] ** 2

# Interaction: PCM (centred) × distance to TFP frontier  → main spec
df['L_pcm_c_x_dist_tfp'] = df['L_pcm_c'] * df['L_dist_tfp_frontier']

# Interaction: PCM (centred) × distance to LP frontier   → robustness
df['L_pcm_c_x_dist_lp']  = df['L_pcm_c'] * df['L_dist_lp_frontier']

# Interaction: OPM (centred) × distance to TFP frontier  → robustness
df['L_opm_c_x_dist_tfp'] = df['L_opm_c'] * df['L_dist_tfp_frontier']

# Squared interaction: PCM² × distance to LP frontier   → robustness (Rob2)
df['L_pcm_c_sq_x_dist_lp'] = df['L_pcm_c_sq'] * df['L_dist_lp_frontier']

# Squared interaction: PCM² × distance to TFP frontier   → main spec (T2)
# Allows the curvature of the PCM–TFP relationship to vary with frontier distance
df['L_pcm_c_sq_x_dist_tfp'] = df['L_pcm_c_sq'] * df['L_dist_tfp_frontier']

# Squared interaction: OPM² × distance to TFP frontier   → robustness (T4)
df['L_opm_c_sq_x_dist_tfp'] = df['L_opm_c_sq'] * df['L_dist_tfp_frontier']

# Industry × year identifier for the second FE dimension
df['nace2_year'] = df['nace2'].astype(str) + '_' + df['year'].astype(str)

# Base controls list (used in all specs)
BASE_CONTROLS = [
    'L_log_employees',
    'L_firm_age',
    'L_leverage',
    'L_capital_intensity',
]

print(f"  Lagged vars created: {len(VARS_TO_LAG)}")
print(f"  Interaction terms  : L_pcm_c_x_dist_tfp, L_pcm_c_sq_x_dist_tfp, "
      f"L_pcm_c_x_dist_lp, L_pcm_c_sq_x_dist_lp, L_opm_c_x_dist_tfp, L_opm_c_sq_x_dist_tfp")
print(f"  Base controls      : {BASE_CONTROLS}")

# Save the fully constructed panel so thesis_tables_figures.py can load it
# without re-running the time-intensive LP estimation.
PANEL_SAVE_PATH = os.path.join(OUTPUT_FOLDER, "panel_analysis.parquet")
df.to_parquet(PANEL_SAVE_PATH, index=False)
print(f"\n  Panel saved to: {PANEL_SAVE_PATH}")
print(f"  Set DATA_PATH in thesis_tables_figures.py to this path.")

# ==============================================================================
'''----Step 12: Descriptive statistics----
We report descriptive statistics for all key variables used in the analysis.
All statistics are at the firm-year level on the cleaned panel before
applying regression-specific sample restrictions.
'''
# ==============================================================================

print("\n[Step 12] Descriptive statistics...")

DESC_VARS = {
    'dlntfp'              : 'TFP growth (ΔlnTFP)',
    'dlnlp'               : 'Labour productivity growth (ΔlnLP)',
    'pcm'                 : 'Price-cost margin (PCM)',
    'opm'                 : 'Operating profit margin (OPM)',
    'dist_tfp_frontier'   : 'Distance to TFP frontier',
    'dist_lp_frontier'    : 'Distance to LP frontier',
    'log_employees'       : 'log(Employees)',
    'firm_age'            : 'Firm age (years)',
    'leverage'            : 'Leverage',
    'capital_intensity'   : 'log(Capital intensity)',
}

desc_rows = []
for var, label in DESC_VARS.items():
    s = df[var].dropna()
    desc_rows.append({
        'Variable'  : label,
        'N'         : int(len(s)),
        'Mean'      : s.mean(),
        'SD'        : s.std(),
        'P10'       : s.quantile(.10),
        'P25'       : s.quantile(.25),
        'Median'    : s.median(),
        'P75'       : s.quantile(.75),
        'P90'       : s.quantile(.90),
        'Min'       : s.min(),
        'Max'       : s.max(),
    })

desc_table = pd.DataFrame(desc_rows).set_index('Variable')
print("\n  Table DS1 — Descriptive Statistics")
print(desc_table.round(3).to_string())

# Sector overview
sector_table = df.groupby('nace2').agg(
    Sector          = ('nace2_label', 'first'),
    Firms           = ('BvD ID number', 'nunique'),
    Obs             = ('year', 'count'),
    Mean_PCM        = ('pcm',             'mean'),
    Mean_dist_TFP   = ('dist_tfp_frontier','mean'),
    Mean_TFP_growth = ('dlntfp',          'mean'),
    Mean_LP_growth  = ('dlnlp',           'mean'),
).sort_values('Firms', ascending=False)

print("\n  Table DS2 — Sector Summary (sorted by firm count)")
print(sector_table.round(3).to_string())

# Save descriptives to CSV
desc_table.round(4).to_csv(
    os.path.join(OUTPUT_FOLDER, 'table_DS1_descriptives.csv')
)
sector_table.round(4).to_csv(
    os.path.join(OUTPUT_FOLDER, 'table_DS2_sectors.csv')
)
betas_table.round(4).to_csv(
    os.path.join(OUTPUT_FOLDER, 'table_PF1_lp_betas.csv')
)
print("\n  Descriptive tables saved to output folder.")


# ==============================================================================
'''----Step 13: Regression helper functions----
Two functions used by all six specifications:

  prepare_sample():
    - Drops observations with missing values in any regression variable.
    - Sets the panel index to (BvD ID number, year) as required by
      linearmodels.
    - Drops singletons (firms with fewer than min_periods observations
      after NA removal) to avoid spurious fixed effects.

  run_twoway_fe():
    - Fits a panel OLS model with:
        (1) Firm (entity) fixed effects — absorbs all time-invariant
            firm heterogeneity.
        (2) Industry × year fixed effects — absorbs common shocks to
            all firms in the same NACE-2 sector in the same year.
    - Industry × year FE are passed via other_effects using integer codes
      (the most efficient approach in linearmodels 7.0).
    - Automatically removes any regressor with zero within-firm variance
      (would otherwise cause a rank failure).
    - Standard errors are clustered at the firm level to account for
      serial correlation within firms.
'''
# ==============================================================================

def prepare_sample(df, dep_var, indep_vars, min_periods=2):
    """
    Prepare regression-ready panel subset.

    Parameters
    ----------
    df           : full panel DataFrame (must contain 'nace2_year' column)
    dep_var      : name of the dependent variable (string)
    indep_vars   : list of regressor names
    min_periods  : minimum firm-year obs after NA removal (default 2)

    Returns
    -------
    sub : DataFrame with panel index (BvD ID number, year)
    """
    needed = [dep_var] + indep_vars + ['nace2_year']
    sub = df.dropna(subset=needed).copy()
    sub = sub.set_index(['BvD ID number', 'year'])
    n_per_firm = sub.groupby(level=0).size()
    valid = n_per_firm[n_per_firm >= min_periods].index
    return sub[sub.index.get_level_values(0).isin(valid)]


def run_twoway_fe(sub, dep_var, indep_vars):
    """
    Panel OLS: firm FE + industry×year FE, firm-clustered SE.

    Parameters
    ----------
    sub        : panel DataFrame with index (BvD ID number, year),
                 must contain 'nace2_year' column
    dep_var    : dependent variable name
    indep_vars : list of regressor names

    Returns
    -------
    linearmodels PanelEffectsResults object
    """
    y = sub[[dep_var]].astype(float)
    X = sub[indep_vars].astype(float)

    # Drop regressors with no within-firm variation (avoid rank failure)
    zero_var_cols = X.var()[X.var() < 1e-10].index.tolist()
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)

    # Industry × year FE as integer codes (other_effects parameter)
    iy_codes = pd.DataFrame(
        {'iy': pd.Categorical(sub['nace2_year']).codes},
        index=sub.index,
    )
    mod = PanelOLS(
        y, X,
        entity_effects=True,     # firm FE
        other_effects=iy_codes,  # industry × year FE
        drop_absorbed=True,
        check_rank=False,        # we handle rank manually above
    )
    return mod.fit(cov_type='clustered', cluster_entity=True)


def _fmt(coef, se, pval):
    """Format a single coefficient with significance stars."""
    stars = ('***' if pval < 0.01 else
             '**'  if pval < 0.05 else
             '*'   if pval < 0.10 else '')
    return f"{coef:+.4f}{stars}", f"({se:.4f})"


# ==============================================================================
'''----Step 14: Run all regression specifications----
We estimate six specifications:

  Spec T1 [Table 1, column 1]: TFP growth ~ PCM + PCM² + controls
    Baseline inverted-U test. Requires α₁ > 0 and α₂ < 0 for an inverted-U.
    This is the replication exercise that contextualises the main result.

  Spec T2 [Table 1, column 2 — MAIN SPECIFICATION]:
    TFP growth ~ PCM + dist(TFP frontier) + PCM×dist + controls
    Marginal effect of PCM = β₁ + β₃ × dist(TFP frontier).
    Hypotheses:
      H1: β₁ > 0  (firms near frontier benefit from market power)
      H2: β₃ < 0  (effect weakens for firms further from frontier)

  Spec 1  [Robustness]: Labour productivity growth ~ PCM + PCM² + controls
  Spec 2  [Robustness]: LP growth ~ PCM + dist(LP frontier) +  dist(LP frontier) + PCM×dist(LP) + PCM²×dist(LP)
  Spec T3 [Robustness]: TFP growth ~ OPM + OPM² + controls
  Spec T4 [Robustness]: TFP growth ~ OPM + dist(TFP frontier) + OPM×dist

All specifications include firm FE + industry×year FE, with standard errors
clustered at the firm level.
'''
# ==============================================================================

print("\n[Step 14] Running regression specifications...")
print("  Estimator : Within estimator (linearmodels PanelOLS)")
print("  FE        : Firm + Industry×Year")
print("  SE        : Clustered at firm level\n")

results = {}   # store all fitted models

# ── Spec T1 — TFP growth, PCM baseline (inverted-U test) ─────────────────────
rhs_T1 = ['L_pcm_c', 'L_pcm_c_sq'] + BASE_CONTROLS
sub_T1  = prepare_sample(df, 'dlntfp', rhs_T1)
res_T1  = run_twoway_fe(sub_T1, 'dlntfp', rhs_T1)
results['T1'] = res_T1

a1 = res_T1.params.get('L_pcm_c',    np.nan)
a2 = res_T1.params.get('L_pcm_c_sq', np.nan)
print(f"  Spec T1 | DV: TFP growth | PCM baseline")
print(f"    N = {int(res_T1.nobs):,}  |  R²(within) = {res_T1.rsquared_within:.4f}  |  "
      f"R²(overall) = {res_T1.rsquared:.4f}")
coef_str, se_str = _fmt(a1, res_T1.std_errors.get('L_pcm_c', np.nan),
                         res_T1.pvalues.get('L_pcm_c', np.nan))
print(f"    PCM (t−1)  : {coef_str}  {se_str}")
coef_str, se_str = _fmt(a2, res_T1.std_errors.get('L_pcm_c_sq', np.nan),
                         res_T1.pvalues.get('L_pcm_c_sq', np.nan))
print(f"    PCM² (t−1) : {coef_str}  {se_str}")
print(f"    Inverted-U (α₁>0 AND α₂<0): "
      f"{'YES ✓' if (a1 > 0 and a2 < 0) else 'NO ✗'}")
if not np.isnan(a2) and a2 != 0:
    peak = -a1 / (2 * a2) + pcm_mean
    print(f"    Implied peak PCM* ≈ {peak:.3f}  (sample mean PCM = {pcm_mean:.3f})")

# ── Spec T2 — TFP growth, PCM + PCM² + frontier + PCM×dist + PCM²×dist ───────
# Full specification: retains the quadratic in PCM established in T1 and allows
# both the linear and quadratic PCM terms to interact with frontier distance.
# Marginal effect of PCM at the average firm (PCM_c = 0):
#   ∂(ΔlnTFP)/∂PCM|_{PCM_c=0} = β₁ + β₄ × dist
# Full marginal effect at any (PCM_c, dist):
#   ∂(ΔlnTFP)/∂PCM = β₁ + 2β₂·PCM_c + β₄·dist + 2β₅·PCM_c·dist
rhs_T2 = [
    'L_pcm_c', 'L_pcm_c_sq',
    'L_dist_tfp_frontier',
    'L_pcm_c_x_dist_tfp', 'L_pcm_c_sq_x_dist_tfp',
] + BASE_CONTROLS
sub_T2  = prepare_sample(df, 'dlntfp', rhs_T2)
res_T2  = run_twoway_fe(sub_T2, 'dlntfp', rhs_T2)
results['T2'] = res_T2

b1 = res_T2.params.get('L_pcm_c',               np.nan)
b2 = res_T2.params.get('L_pcm_c_sq',             np.nan)
b3 = res_T2.params.get('L_dist_tfp_frontier',    np.nan)
b4 = res_T2.params.get('L_pcm_c_x_dist_tfp',    np.nan)
b5 = res_T2.params.get('L_pcm_c_sq_x_dist_tfp', np.nan)
print(f"\n  Spec T2 | DV: TFP growth | PCM + PCM² + frontier interactions  *** MAIN SPEC ***")
print(f"    N = {int(res_T2.nobs):,}  |  R²(within) = {res_T2.rsquared_within:.4f}  |  "
      f"R²(overall) = {res_T2.rsquared:.4f}")
for v, lbl in [
    ('L_pcm_c',               'PCM (t−1)'),
    ('L_pcm_c_sq',             'PCM² (t−1)'),
    ('L_dist_tfp_frontier',   'Dist(TFP front.) (t−1)'),
    ('L_pcm_c_x_dist_tfp',   'PCM × Dist(TFP) (t−1)'),
    ('L_pcm_c_sq_x_dist_tfp','PCM² × Dist(TFP) (t−1)'),
]:
    if v in res_T2.params.index:
        c_str, s_str = _fmt(res_T2.params[v], res_T2.std_errors[v],
                             res_T2.pvalues[v])
        print(f"    {lbl:<36}: {c_str}  {s_str}")
print(f"    H1 (β₁ < 0 at average firm): "
      f"{'SUPPORTED ✓' if b1 < 0 else 'NOT SUPPORTED ✗'}  "
      f"p = {res_T2.pvalues.get('L_pcm_c', np.nan):.3f}")
print(f"    H2 (β₄ > 0, effect weakens with distance): "
      f"{'SUPPORTED ✓' if b4 > 0 else 'NOT SUPPORTED ✗'}  "
      f"p = {res_T2.pvalues.get('L_pcm_c_x_dist_tfp', np.nan):.3f}")

# ── Spec 1 — Labour productivity growth, PCM baseline (robustness) ────────────
rhs_1 = ['L_pcm_c', 'L_pcm_c_sq'] + BASE_CONTROLS
sub_1  = prepare_sample(df, 'dlnlp', rhs_1)
res_1  = run_twoway_fe(sub_1, 'dlnlp', rhs_1)
results['1'] = res_1

c1 = res_1.params.get('L_pcm_c',    np.nan)
c2 = res_1.params.get('L_pcm_c_sq', np.nan)
print(f"\n  Spec 1  | DV: Labour productivity growth | PCM baseline [robustness]")
print(f"    N = {int(res_1.nobs):,}  |  R²(within) = {res_1.rsquared_within:.4f}  |  "
      f"R²(overall) = {res_1.rsquared:.4f}")
for v, lbl in [('L_pcm_c', 'PCM (t−1)'), ('L_pcm_c_sq', 'PCM² (t−1)')]:
    if v in res_1.params.index:
        c_str, s_str = _fmt(res_1.params[v], res_1.std_errors[v],
                             res_1.pvalues[v])
        print(f"    {lbl:<32}: {c_str}  {s_str}")
print(f"    Inverted-U: {'YES ✓' if (c1 > 0 and c2 < 0) else 'NO ✗'}")

# ── Spec 2 — Labour productivity growth, PCM × LP-frontier (robustness) ───────
rhs_2 = ['L_pcm_c', 'L_pcm_c_sq', 'L_dist_lp_frontier', 'L_pcm_c_x_dist_lp', 'L_pcm_c_sq_x_dist_lp'] + BASE_CONTROLS
sub_2  = prepare_sample(df, 'dlnlp', rhs_2)
res_2  = run_twoway_fe(sub_2, 'dlnlp', rhs_2)
results['2'] = res_2

d1 = res_2.params.get('L_pcm_c',           np.nan)
d3 = res_2.params.get('L_pcm_c_x_dist_lp', np.nan)
print(f"\n  Spec 2  | DV: Labour productivity growth | PCM × LP-frontier [robustness]")
print(f"    N = {int(res_2.nobs):,}  |  R²(within) = {res_2.rsquared_within:.4f}  |  "
      f"R²(overall) = {res_2.rsquared:.4f}")
for v, lbl in [('L_pcm_c',                'PCM (t−1)'),
               ('L_pcm_c_sq',             'PCM² (t−1)'),
               ('L_dist_lp_frontier',     'Dist(LP front.) (t−1)'),
               ('L_pcm_c_x_dist_lp',      'PCM × Dist(LP) (t−1)'),
               ('L_pcm_c_sq_x_dist_lp',  'PCM² × Dist(LP) (t−1)')]:
    if v in res_2.params.index:
        c_str, s_str = _fmt(res_2.params[v], res_2.std_errors[v],
                             res_2.pvalues[v])
        print(f"    {lbl:<32}: {c_str}  {s_str}")

# ── Spec T3 — TFP growth, OPM baseline (robustness) ──────────────────────────
rhs_T3 = ['L_opm_c', 'L_opm_c_sq'] + BASE_CONTROLS
sub_T3  = prepare_sample(df, 'dlntfp', rhs_T3)
res_T3  = run_twoway_fe(sub_T3, 'dlntfp', rhs_T3)
results['T3'] = res_T3

print(f"\n  Spec T3 | DV: TFP growth | OPM baseline [robustness]")
print(f"    N = {int(res_T3.nobs):,}  |  R²(within) = {res_T3.rsquared_within:.4f}  |  "
      f"R²(overall) = {res_T3.rsquared:.4f}")
for v, lbl in [('L_opm_c', 'OPM (t−1)'), ('L_opm_c_sq', 'OPM² (t−1)')]:
    if v in res_T3.params.index:
        c_str, s_str = _fmt(res_T3.params[v], res_T3.std_errors[v],
                             res_T3.pvalues[v])
        print(f"    {lbl:<32}: {c_str}  {s_str}")

# ── Spec T4 — TFP growth, OPM + OPM² + frontier + OPM×dist + OPM²×dist ──────
rhs_T4 = [
    'L_opm_c', 'L_opm_c_sq',
    'L_dist_tfp_frontier',
    'L_opm_c_x_dist_tfp', 'L_opm_c_sq_x_dist_tfp',
] + BASE_CONTROLS
sub_T4  = prepare_sample(df, 'dlntfp', rhs_T4)
res_T4  = run_twoway_fe(sub_T4, 'dlntfp', rhs_T4)
results['T4'] = res_T4

print(f"\n  Spec T4 | DV: TFP growth | OPM + OPM² + frontier interactions [robustness]")
print(f"    N = {int(res_T4.nobs):,}  |  R²(within) = {res_T4.rsquared_within:.4f}  |  "
      f"R²(overall) = {res_T4.rsquared:.4f}")
for v, lbl in [
    ('L_opm_c',               'OPM (t−1)'),
    ('L_opm_c_sq',             'OPM² (t−1)'),
    ('L_dist_tfp_frontier',   'Dist(TFP front.) (t−1)'),
    ('L_opm_c_x_dist_tfp',   'OPM × Dist(TFP) (t−1)'),
    ('L_opm_c_sq_x_dist_tfp','OPM² × Dist(TFP) (t−1)'),
]:
    if v in res_T4.params.index:
        c_str, s_str = _fmt(res_T4.params[v], res_T4.std_errors[v],
                             res_T4.pvalues[v])
        print(f"    {lbl:<36}: {c_str}  {s_str}")


# ==============================================================================
'''----Step 15: Marginal effects of PCM across the TFP-frontier distribution----
For the main specification (Spec T2), the marginal effect of the price-cost
margin on TFP growth is not a single number — it varies with each firm's
distance to the TFP frontier:

  ∂(ΔlnTFP) / ∂PCM = β₁ + β₃ × dist_tfp

We evaluate this marginal effect at five percentiles of dist_tfp_frontier
(P10, P25, P50, P75, P90) and compute standard errors via the delta method:

  SE(ME) = sqrt[ SE(β₁)² + dist² × SE(β₃)² + 2 × dist × Cov(β₁, β₃) ]

p-values are computed under the two-tailed normal approximation.
'''
# ==============================================================================

print("\n[Step 15] Marginal effects of PCM on TFP growth (Spec T2)...")
print(f"  Full ME: ∂(ΔlnTFP)/∂PCM = β₁ + 2β₂·PCM_c + β₄·dist + 2β₅·PCM_c·dist")
print(f"  Evaluated at PCM_c = 0 (sample mean, since PCM is centred):")
print(f"  ME|_{{PCM_c=0}} = β₁ + β₄ × dist")
print(f"  β₁ = {b1:+.4f}  |  β₄ = {b4:+.4f}\n")

me_rows = []
se_b1 = res_T2.std_errors.get('L_pcm_c',            np.nan)
se_b4 = res_T2.std_errors.get('L_pcm_c_x_dist_tfp', np.nan)
cov_b1_b4 = (
    res_T2.cov.loc['L_pcm_c', 'L_pcm_c_x_dist_tfp']
    if ('L_pcm_c' in res_T2.cov.index and
        'L_pcm_c_x_dist_tfp' in res_T2.cov.index)
    else 0.0
)

print(f"  {'Percentile':<26} {'dist value':>11} {'ME of PCM':>11} "
      f"{'SE':>8} {'p-value':>9} {'Sig.':>5}")
print("  " + "-" * 74)

for pct_label, pct in [
    ('P10 — near frontier',    .10),
    ('P25',                    .25),
    ('P50 — median',           .50),
    ('P75',                    .75),
    ('P90 — far from frontier', .90),
]:
    d    = df['L_dist_tfp_frontier'].quantile(pct)
    if np.isnan(d):
        continue
    # ME at the average firm (PCM_c = 0): β₁ + β₄·d
    me   = b1 + b4 * d
    se_me = np.sqrt(
        se_b1 ** 2
        + d ** 2 * se_b4 ** 2
        + 2 * d * cov_b1_b4
    )
    t_stat = me / se_me if se_me > 0 else np.nan
    p_val  = 2 * (1 - norm.cdf(abs(t_stat)))
    sig    = ('***' if p_val < 0.01 else
              '**'  if p_val < 0.05 else
              '*'   if p_val < 0.10 else '')
    print(f"  {pct_label:<26} {d:>11.3f} {me:>+11.4f} "
          f"{se_me:>8.4f} {p_val:>9.3f} {sig:>5}")
    me_rows.append({'Percentile': pct_label, 'dist_TFP_frontier': round(d, 3),
                    'ME_PCM': round(me, 4), 'SE': round(se_me, 4),
                    'p_value': round(p_val, 3)})

me_table = pd.DataFrame(me_rows).set_index('Percentile')
me_table.to_csv(os.path.join(OUTPUT_FOLDER, 'table_ME1_marginal_effects.csv'))
print("\n  Marginal effects table saved.")


# ==============================================================================
'''----Step 16: Print formatted regression tables----
Two formatted tables:

  Table R1 — Main and robustness results (all 6 specifications).
    - Rows: all regressors appearing in at least one spec.
    - Columns: Spec T1, T2 (main), 1, 2, T3, T4.
    - Entries: coefficient with significance stars (top row) and
               standard error in parentheses (bottom row).

  Table PF1 — Levinsohn-Petrin production function estimates.
    - Rows: NACE-2 sectors.
    - Columns: N_obs, β_labour, β_capital, returns to scale (β_l + β_k).

Both tables are also saved as CSV to the output folder.
'''
# ==============================================================================

print("\n[Step 16] Formatted regression tables...")

ALL_SPECS_ORDERED = [
    ('T1', 'TFP growth\nPCM base',    results['T1']),
    ('T2', 'TFP growth\nPCM×dist(TFP)\n[MAIN]', results['T2']),
    ('1',  'LP growth\nPCM base',     results['1']),
    ('2',  'LP growth\nPCM×dist(LP)', results['2']),
    ('T3', 'TFP growth\nOPM base',    results['T3']),
    ('T4', 'TFP growth\nOPM×dist(TFP)', results['T4']),
]

DISPLAY_VARS_ORDER = [
    ('L_pcm_c',                  'PCM (centred, t−1)'),
    ('L_pcm_c_sq',               'PCM² (centred, t−1)'),
    ('L_opm_c',                  'OPM (centred, t−1)'),
    ('L_opm_c_sq',               'OPM² (centred, t−1)'),
    ('L_dist_tfp_frontier',      'Dist. to TFP frontier (t−1)'),
    ('L_dist_lp_frontier',       'Dist. to LP frontier (t−1)'),
    ('L_pcm_c_x_dist_lp',        'PCM × Dist(LP) (t−1)'),
    ('L_pcm_c_x_dist_tfp',       'PCM × Dist(TFP) (t−1)'),
    ('L_pcm_c_sq_x_dist_tfp',   'PCM² × Dist(TFP) (t−1)'),
    ('L_pcm_c_sq_x_dist_lp',    'PCM² × Dist(LP) (t−1)'),
    ('L_opm_c_x_dist_tfp',       'OPM × Dist(TFP) (t−1)'),
    ('L_opm_c_sq_x_dist_tfp',   'OPM² × Dist(TFP) (t−1)'),
    ('L_log_employees',          'log(Employees) (t−1)'),
    ('L_leverage',               'Leverage (t−1)'),
    ('L_capital_intensity',      'Capital intensity (t−1)'),
    ('L_firm_age',               'Firm age (t−1)'),
]

W = 18    # column width

# Header
header = f"{'Variable':<36}" + "".join(
    f"Spec {sid}".rjust(W) for sid, _, _ in ALL_SPECS_ORDERED
)
sep = "=" * (36 + W * len(ALL_SPECS_ORDERED))

print(f"\n\n  Table R1 — Regression Results")
print(f"  Dependent variables: TFP growth (ΔlnTFP) or Labour productivity growth (ΔlnLP)")
print(sep)
print(header)
print("-" * (36 + W * len(ALL_SPECS_ORDERED)))

table_rows = []   # for CSV export

for vname, vlabel in DISPLAY_VARS_ORDER:
    coef_row = f"{vlabel:<36}"
    se_row   = f"{'':36}"
    csv_row  = {'Variable': vlabel}
    present  = False

    for sid, _, res in ALL_SPECS_ORDERED:
        if vname in res.params.index:
            c  = res.params[vname]
            s  = res.std_errors[vname]
            p  = res.pvalues[vname]
            st = ('***' if p < 0.01 else '**' if p < 0.05
                  else '*' if p < 0.10 else '')
            coef_row += f"{c:+.4f}{st:>3}".rjust(W)
            se_row   += f"({s:.4f})".rjust(W)
            csv_row[f'Spec{sid}_coef'] = round(c, 4)
            csv_row[f'Spec{sid}_se']   = round(s, 4)
            csv_row[f'Spec{sid}_p']    = round(p, 3)
            present = True
        else:
            coef_row += f"{'—':>{W}}"
            se_row   += f"{'':>{W}}"

    if present:
        print(coef_row)
        print(se_row)
        table_rows.append(csv_row)

print("-" * (36 + W * len(ALL_SPECS_ORDERED)))

# Footer rows
for label, extractor in [
    ('Observations',  lambda r: f"{int(r.nobs):,}"),
    ('R² (within)',   lambda r: f"{r.rsquared_within:.4f}"),
    ('R² (overall)',  lambda r: f"{r.rsquared:.4f}"),
    ('Firm FE',       lambda r: 'YES'),
    ('Industry×Year FE', lambda r: 'YES'),
]:
    row = f"{label:<36}"
    for _, _, res in ALL_SPECS_ORDERED:
        row += extractor(res).rjust(W)
    print(row)

print(sep)
print("  * p<0.10  ** p<0.05  *** p<0.01")
print("  Standard errors in parentheses, clustered at firm level.")
print("  All specifications include firm FE and industry×year FE.")
print("  PCM and OPM centred at their sample means to reduce multicollinearity.")
print("  LP  = labour productivity (revenue per employee). Single-factor measure.")
print("  TFP = total factor productivity (Levinsohn-Petrin, 2003). Multi-factor measure.")
print("  These are distinct measures and must not be conflated.")
print("  Export dummy omitted: export revenue not available in this ORBIS extract.")

# Save Table R1
pd.DataFrame(table_rows).to_csv(
    os.path.join(OUTPUT_FOLDER, 'table_R1_regressions.csv'), index=False
)

# Print Table PF1
print(f"\n\n  Table PF1 — Levinsohn-Petrin Production Function Estimates by Sector")
print(f"  (Value-added production function: ln(VA) = β_l·ln(L) + β_k·ln(K) + ω)")
print("  " + "-" * 70)
print(f"  {'NACE':>5}  {'Sector':<28}  {'N obs':>7}  "
      f"{'β_labour':>9}  {'β_capital':>10}  {'RTS':>6}")
print("  " + "-" * 70)
for _, row in betas_table.iterrows():
    print(f"  {row.name:>5}  {str(row['label'])[:27]:<28}  "
          f"{int(row['N_obs']):>7,}  "
          f"{row['beta_l']:>+9.3f}  {row['beta_k']:>+10.3f}  "
          f"{row['RTS']:>6.3f}")
print("  " + "-" * 70)
print("  RTS = Returns to scale = β_labour + β_capital.")
print("  β_labour and β_capital are elasticities of value added with respect to")
print("  labour and capital respectively, estimated via the LP two-stage GMM procedure.")

print("\n\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print(f"Output saved to: {OUTPUT_FOLDER}")
print("=" * 70)



"""
================================================================================
SCATTER PLOT — PCM vs TFP Growth by Frontier Distance Tercile
Market Power and Innovation in Spanish Manufacturing Firms

Run this script after 0501_RegressionAnalysis_v03.py has completed.
It loads the saved panel_analysis.parquet file and produces two plots:

  Plot 1 — Scatter with binned means and LOWESS smoother
    PCM on x-axis, TFP growth on y-axis, coloured by frontier distance tercile.
    Raw firm-year observations are shown as faint points; binned means are
    overlaid as larger markers with a LOWESS smoother through each tercile.
    This is the clearest way to see the shape of the relationship.

  Plot 2 — Marginal effects plot
    The marginal effect of PCM on TFP growth (from Spec T2) evaluated across
    the continuous distribution of frontier distance, with a 95% confidence
    band. Shows clearly where the effect is significant and where it fades.

OUTPUT: both plots saved as high-resolution PNGs to OUTPUT_FOLDER.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess

# ── CHANGE THIS to match your OUTPUT_FOLDER from the main script ──────────────
OUTPUT_FOLDER = r"C:\Users\anasr\Desktop\TFGS\TFG ECONOMIA 2\02. Data\01. Outputs"
PANEL_PATH    = os.path.join(OUTPUT_FOLDER, "panel_analysis.parquet")

# ── T2 coefficients — paste from your Step 14 output ─────────────────────────
# These come directly from the v03 regression output
BETA_1 = -0.1763   # PCM (centred)
BETA_4 = +0.1230   # PCM × Dist(TFP frontier)
SE_B1  =  0.0458
SE_B4  =  0.0434
COV_B1_B4 = -0.0003   # update if you have the exact covariance; small either way

# ── Plotting style ─────────────────────────────────────────────────────────────
COLORS = {
    'Near frontier (T1)'   : '#2166ac',   # dark blue
    'Middle (T2)'          : '#74add1',   # mid blue
    'Far frontier (T3)'    : '#d73027',   # red
}
ALPHA_RAW  = 0.04   # transparency of raw scatter points
ALPHA_BIN  = 0.85   # transparency of binned means
FSIZE      = 11

plt.rcParams.update({
    'font.family'     : 'serif',
    'font.size'       : FSIZE,
    'axes.spines.top' : False,
    'axes.spines.right': False,
})

# ==============================================================================
# Load data
# ==============================================================================
print("Loading panel data...")
df = pd.read_parquet(PANEL_PATH)

# Use lagged PCM (centred) and lagged frontier distance — same as regressions
plot_df = df[['L_pcm_c', 'dlntfp', 'L_dist_tfp_frontier']].dropna().copy()

# Winsorise for plotting (already done in script but parquet keeps raw lags)
for col in ['L_pcm_c', 'dlntfp', 'L_dist_tfp_frontier']:
    lo = plot_df[col].quantile(0.01)
    hi = plot_df[col].quantile(0.99)
    plot_df[col] = plot_df[col].clip(lo, hi)

# Terciles of frontier distance
plot_df['tercile'] = pd.qcut(
    plot_df['L_dist_tfp_frontier'], q=3,
    labels=['Near frontier (T1)', 'Middle (T2)', 'Far frontier (T3)']
)

print(f"  Observations for plotting: {len(plot_df):,}")
print(f"  Tercile counts:\n{plot_df['tercile'].value_counts().sort_index()}")

# ==============================================================================
# Plot 1 — Scatter with binned means and LOWESS
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

for label, color in COLORS.items():
    sub = plot_df[plot_df['tercile'] == label]

    # Raw scatter (very transparent)
    ax.scatter(
        sub['L_pcm_c'], sub['dlntfp'],
        color=color, alpha=ALPHA_RAW, s=4, linewidths=0, rasterized=True
    )

    # Binned means (20 equal-width bins of PCM within tercile)
    sub = sub.copy()
    sub['pcm_bin'] = pd.cut(sub['L_pcm_c'], bins=20)
    binned = sub.groupby('pcm_bin', observed=True).agg(
        pcm_mid  = ('L_pcm_c', 'mean'),
        tfp_mean = ('dlntfp',  'mean'),
        n        = ('dlntfp',  'count'),
    ).dropna()
    binned = binned[binned['n'] >= 30]   # only show bins with enough obs

    ax.scatter(
        binned['pcm_mid'], binned['tfp_mean'],
        color=color, alpha=ALPHA_BIN, s=45, zorder=3,
        edgecolors='white', linewidths=0.5
    )

    # LOWESS smoother through binned means
    if len(binned) >= 5:
        smoothed = lowess(
            binned['tfp_mean'], binned['pcm_mid'],
            frac=0.6, return_sorted=True
        )
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, lw=2, zorder=4)

# Reference lines
ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
ax.axvline(0, color='grey', lw=0.8, ls='--', alpha=0.5)

# Legend
patches = [
    mpatches.Patch(color=c, label=l) for l, c in COLORS.items()
]
ax.legend(
    handles=patches, title='Frontier distance tercile',
    fontsize=9, title_fontsize=9,
    framealpha=0.9, loc='upper right'
)

ax.set_xlabel('PCM (centred at sample mean)', fontsize=FSIZE)
ax.set_ylabel('TFP growth ($\\Delta \\ln$ TFP)', fontsize=FSIZE)
ax.set_title(
    'PCM and TFP Growth by Distance to the Technological Frontier',
    fontsize=FSIZE + 1, pad=10
)
ax.text(
    0.01, 0.01,
    'Points: binned means (≥30 obs). Lines: LOWESS smoother. '
    'Raw observations shown as faint background.',
    transform=ax.transAxes, fontsize=7, color='grey', va='bottom'
)

plt.tight_layout()
out1 = os.path.join(OUTPUT_FOLDER, 'fig_scatter_pcm_tfp_terciles.png')
fig.savefig(out1, dpi=300, bbox_inches='tight')
print(f"\n  Plot 1 saved: {out1}")
plt.close()

# ==============================================================================
# Plot 2 — Marginal effects of PCM across frontier distance distribution
# ==============================================================================
d_grid = np.linspace(
    plot_df['L_dist_tfp_frontier'].quantile(0.05),
    plot_df['L_dist_tfp_frontier'].quantile(0.95),
    200
)

me   = BETA_1 + BETA_4 * d_grid
se   = np.sqrt(SE_B1**2 + d_grid**2 * SE_B4**2 + 2 * d_grid * COV_B1_B4)
ci_lo = me - 1.96 * se
ci_hi = me + 1.96 * se

# Significance threshold — where does ME cross zero?
zero_cross = d_grid[np.where(np.diff(np.sign(me)))[0]]

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.fill_between(d_grid, ci_lo, ci_hi, alpha=0.15, color='#2166ac', label='95% CI')
ax.plot(d_grid, me, color='#2166ac', lw=2, label='Marginal effect of PCM')
ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.6)

# Mark the five percentile points from the marginal effects table
me_table = {
    'P10': (0.198, -0.1519),
    'P25': (0.494, -0.1154),
    'P50': (0.787, -0.0794),
    'P75': (1.063, -0.0455),
    'P90': (1.324, -0.0134),
}
for pct, (d_val, me_val) in me_table.items():
    sig = pct in ('P10', 'P25', 'P50', 'P75')
    ax.scatter(
        d_val, me_val,
        color='#2166ac' if sig else 'white',
        edgecolors='#2166ac', s=60, zorder=5, linewidths=1.5
    )
    ax.annotate(
        pct, xy=(d_val, me_val),
        xytext=(5, -12), textcoords='offset points',
        fontsize=8, color='#333333'
    )

# Shade the region where effect is insignificant (roughly P90+)
if len(zero_cross) > 0:
    ax.axvspan(zero_cross[0], d_grid[-1], alpha=0.06, color='grey',
               label='Effect not sig. at P90')

ax.set_xlabel('Distance to TFP frontier (lagged)', fontsize=FSIZE)
ax.set_ylabel('$\\partial(\\Delta \\ln TFP) / \\partial PCM$', fontsize=FSIZE)
ax.set_title(
    'Marginal Effect of PCM on TFP Growth across the Frontier Distribution',
    fontsize=FSIZE + 1, pad=10
)
ax.legend(fontsize=9, framealpha=0.9)
ax.text(
    0.01, 0.01,
    'Marginal effect evaluated at PCM$_c$ = 0 (sample mean). '
    'Filled points: significant at $p < 0.05$. Open point: insignificant (P90).',
    transform=ax.transAxes, fontsize=7, color='grey', va='bottom'
)

plt.tight_layout()
out2 = os.path.join(OUTPUT_FOLDER, 'fig_marginal_effects_pcm.png')
fig.savefig(out2, dpi=300, bbox_inches='tight')
print(f"  Plot 2 saved: {out2}")
plt.close()

print("\nDone. Both figures saved to output folder.")
print("Required packages: matplotlib, scipy, statsmodels, pandas, numpy")
print("Install if needed: pip install statsmodels")



# Mean TFP growth by PCM tercile and frontier distance tercile
df['pcm_tercile'] = pd.qcut(df['pcm'], q=3, 
                             labels=['Low PCM', 'Medium PCM', 'High PCM'])
df['dist_tercile'] = pd.qcut(df['dist_tfp_frontier'], q=3,
                              labels=['Near frontier', 'Middle', 'Far frontier'])

tercile_means = df.groupby(['dist_tercile', 'pcm_tercile'])['dlntfp'].mean().unstack()
print(tercile_means.round(3))