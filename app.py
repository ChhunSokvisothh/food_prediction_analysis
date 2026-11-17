import io
import itertools
import json
import logging
import os
import re
from difflib import get_close_matches

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from telegram import BotCommand, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# --- env ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")
ADMIN_ID = int(os.getenv("ADMIN_ID"))
MODELS_DIR = os.getenv("MODELS_DIR", "models")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("foodbot")


# =============================================================================
# Loading models
#
# OLD  = the models the bot originally shipped with (never overwritten):
#          price_forecaster.pkl, margin_classifier.pkl, national-average anomaly stats
# NEW  = the improved versions from the notebook. Every one of these is OPTIONAL:
#          price_forecaster_improved.pkl (+ _features.pkl)   notebook cell 34
#          province_stats.pkl, margin_history.pkl            notebook cell 56
#          benchmark.json, reg_<key>.pkl, clf_<key>.pkl      notebook cell 55
#        Whatever is missing silently falls back to the OLD behaviour, and
#        /model shows exactly what is and isn't active.
# =============================================================================
def _path(name):
    return os.path.join(MODELS_DIR, name)


def load_required(name):
    return joblib.load(_path(name))


def load_optional(name):
    path = _path(name)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:  # a bad optional file must never stop the bot starting
        log.warning("Could not load %s: %s", name, e)
        return None


def load_optional_json(name):
    path = _path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not read %s: %s", name, e)
        return None


# shared by every version
iso = load_required("anomaly_detector.pkl")
scaler = load_required("anomaly_scaler.pkl")
commodity_to_code = load_required("commodity_encoder.pkl")
province_to_code = load_required("province_encoder.pkl")
reference_stats = load_required("reference_stats.pkl")

COMMODITIES = sorted(commodity_to_code.keys())
PROVINCES = sorted(province_to_code.keys())

BENCH = load_optional_json("benchmark.json")  # written by notebook cell 55

# --- price forecasters -------------------------------------------------------
# kind "level"  : model predicts the price directly           (original XGBoost)
# kind "change" : model predicts log(price / last month's price); the price is
#                 then  lag_1 * exp(prediction)               (improved XGBoost)
FEATURES_LEVEL = [
    'lag_1', 'lag_2', 'lag_3', 'rolling_3m', 'rolling_6m', 'price_momentum',
    'month', 'quarter', 'year', 'is_wet_season', 'is_retail',
    'province_enc', 'commodity_enc',
]
FEATURES_CHANGE = load_optional("price_forecaster_improved_features.pkl") or [
    'r_lag2', 'r_lag3', 'r_roll3', 'r_roll6', 'r_momentum', 'lag_1',
    'month', 'quarter', 'is_wet_season', 'is_retail',
    'province_enc', 'commodity_enc',
]

REG_MODELS = {}


def register_reg(key, name, kind, model):
    REG_MODELS[key] = {"name": name, "kind": kind, "model": model}


register_reg("v1", "Original XGBoost", "level", load_required("price_forecaster.pkl"))

_m = load_optional("price_forecaster_improved.pkl")
if _m is not None:
    register_reg("v2", "Improved XGBoost", "change", _m)

for _key, _meta in (BENCH or {}).get("regression", {}).items():
    _m = load_optional(f"reg_{_key}.pkl")
    if _m is not None and _meta.get("kind") in ("level", "change"):
        register_reg(_key, _meta.get("name", _key), _meta["kind"], _m)

_default_reg = (BENCH or {}).get("default_reg")
NEW_REG_KEY = _default_reg if _default_reg in REG_MODELS else ("v2" if "v2" in REG_MODELS else "v1")

# --- margin classifiers ------------------------------------------------------
CLF_MODELS = {"v1": {"name": "Gradient Boosting (original)", "model": load_required("margin_classifier.pkl")}}

for _key, _meta in (BENCH or {}).get("classification", {}).items():
    _m = load_optional(f"clf_{_key}.pkl")
    if _m is not None:
        CLF_MODELS[_key] = {"name": _meta.get("name", _key), "model": _m}

_default_clf = (BENCH or {}).get("default_clf")
NEW_CLF_KEY = _default_clf if _default_clf in CLF_MODELS else "v1"

MARGIN_FEATURES = list(getattr(CLF_MODELS["v1"]["model"], "feature_names_in_", [
    'commodity_enc', 'market_enc', 'month', 'year', 'is_wet_season',
    'wholesale_price', 'lag_margin', 'lag_margin_2', 'rolling_margin',
]))
MEDIAN_MARGIN = 11.1  # median retail-over-wholesale margin (%), from the notebook

# --- optional extra stats for the NEW version --------------------------------
# province_stats : (province, commodity) -> usdprice, lag_1, lag_2, rolling_3m  (retail)
# margin_history : commodity -> last 6 observed margin % values
PROVINCE_STATS = load_optional("province_stats.pkl")
MARGIN_HISTORY = load_optional("margin_history.pkl") or {}

ANOM_FEATURES = list(getattr(scaler, "feature_names_in_", [
    'usdprice', 'lag_1', 'price_momentum', 'rolling_3m', 'is_wet_season', 'month',
]))

# What is actually improved in "new"?
NEW_PRICE = NEW_REG_KEY != "v1"
NEW_MARGIN_MODEL = NEW_CLF_KEY != "v1"
NEW_MARGIN_INPUTS = bool(MARGIN_HISTORY)
NEW_ANOMALY = PROVINCE_STATS is not None
HAS_NEW = NEW_PRICE or NEW_MARGIN_MODEL or NEW_MARGIN_INPUTS or NEW_ANOMALY
DEFAULT_VERSION = "new" if HAS_NEW else "old"

VERSION_LABEL = {"old": "🕰️ OLD", "new": "🆕 NEW"}

# Scores copied from the notebook's executed outputs (sections 10.1 / 10.2 / 10.3,
# date-based split). Only used by /benchmark when models/benchmark.json is absent.
FALLBACK_BENCH = {
    "baseline_mae": 0.1734,
    "price": [
        ("OLD  Original XGBoost", 0.1810, -4.4),
        ("NEW  Improved XGBoost", 0.1598, 7.9),
    ],
    "margin_baseline_acc": 0.798,
    "margin": [("OLD  Gradient Boosting", 0.843, 0.906)],
}

# alert storage (in memory, resets on every restart / sleep)
alerts = {}


# =============================================================================
# Helpers
# =============================================================================
def get_month_features(month):
    return {
        'month': month,
        'quarter': (month - 1) // 3 + 1,
        'year': 2026,  # kept as-is: the OLD model was trained with a `year` feature
        'is_wet_season': 1 if month in [5, 6, 7, 8, 9, 10] else 0
    }


def fuzzy_match(name, options):
    name = name.lower().strip()
    for opt in options:
        if name in opt.lower() or opt.lower() in name:
            return opt
    close = get_close_matches(name, [o.lower() for o in options], n=3, cutoff=0.7)  # strict: a typo should ask 'did you mean', not guess
    if close:
        for opt in options:
            if opt.lower() == close[0]:
                return opt
    return None


def fuzzy_suggestions(name, options, n=3):
    name = name.lower().strip()
    matches = get_close_matches(name, [o.lower() for o in options], n=n, cutoff=0.3)
    suggestions = []
    for m in matches:
        for opt in options:
            if opt.lower() == m:
                suggestions.append(opt)
    return suggestions


def price_trend_emoji(change_pct):
    if abs(change_pct) <= 5:
        return "🟢", "Stable"
    elif abs(change_pct) <= 15:
        return "🟡", "Moderate change"
    else:
        return "🔴", "Significant spike" if change_pct > 0 else "Significant drop"


def fit(label, width):
    """Truncate with an ellipsis so long model names don't get cut mid-word in tables."""
    return label if len(label) <= width else label[:width - 1] + "…"


def get_version(context):
    return context.user_data.get("version", DEFAULT_VERSION)


def reg_key_for(version):
    return NEW_REG_KEY if version == "new" else "v1"


def clf_key_for(version):
    return NEW_CLF_KEY if version == "new" else "v1"


def reg_name_for(version):
    return REG_MODELS[reg_key_for(version)]["name"]


def model_footer(version):
    return f"🧠 Model: `{reg_name_for(version)}` ({VERSION_LABEL[version]})"


# =============================================================================
# Prediction engine
# =============================================================================
def build_reg_row(kind, province, commodity, pricetype, month,
                  lag_1, lag_2, lag_3, rolling_3m, rolling_6m):
    """One-row DataFrame with exactly the columns (and order) the model was trained on."""
    mf = get_month_features(month)
    momentum = lag_1 - lag_2
    row = {
        'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3,
        'rolling_3m': rolling_3m, 'rolling_6m': rolling_6m,
        'price_momentum': momentum,
        'month': mf['month'], 'quarter': mf['quarter'], 'year': mf['year'],
        'is_wet_season': mf['is_wet_season'],
        'is_retail': 1 if pricetype == 'Retail' else 0,
        'province_enc': province_to_code[province],
        'commodity_enc': commodity_to_code[commodity],
    }
    if kind == "change":
        # same formulas as add_relative_features() in the notebook
        row.update({
            'r_lag2': lag_2 / lag_1, 'r_lag3': lag_3 / lag_1,
            'r_roll3': rolling_3m / lag_1, 'r_roll6': rolling_6m / lag_1,
            'r_momentum': momentum / lag_1,
        })
        return pd.DataFrame([row])[FEATURES_CHANGE]
    return pd.DataFrame([row])[FEATURES_LEVEL]


def predict_from_lags(key, province, commodity, pricetype, month,
                      lag_1, lag_2, lag_3, rolling_3m, rolling_6m):
    entry = REG_MODELS[key]
    X = build_reg_row(entry["kind"], province, commodity, pricetype, month,
                      lag_1, lag_2, lag_3, rolling_3m, rolling_6m)
    raw = float(entry["model"].predict(X)[0])
    if entry["kind"] == "change":
        return lag_1 * float(np.exp(raw))  # predicted % change -> price
    return raw


HORIZONS = (1, 2, 3, 6, 12)    # months ahead offered for price forecasts
CHART_HORIZONS = (3, 6, 12)    # how far a chart can look
MAX_MONTHS = 12


def horizon_label(months):
    target = pd.Timestamp.now() + pd.DateOffset(months=months)
    when = "next month" if months == 1 else f"{months} months ahead"
    return f"{target.strftime('%B %Y')} ({when})"


def parse_months(text):
    """'3', '3m', '3 months' -> 3 (only 1..12 are valid), anything else -> None."""
    m = re.fullmatch(r"(\d{1,2})\s*(m|mo|mos|month|months)?", text.strip().lower())
    if m and 1 <= int(m.group(1)) <= MAX_MONTHS:
        return int(m.group(1))
    return None


LONG_RANGE_NOTE = ("⚠️ Only 1-month-ahead accuracy was tested. Beyond that the model feeds its own "
                   "forecasts back in, so errors can compound.")


def forecast_prices(key, commodity, province, pricetype, months=1):
    """Forecast 1..`months` steps ahead from the stored averages; returns one price per month.

    Matches how the models were trained: each training row paired the PREVIOUS month's price
    (lag_1) with the TARGET month's calendar. So step 1 uses lag_1 = the latest price and
    month = next month; every later step feeds the previous forecast back in as lag_1.
    """
    st = reference_stats.loc[(commodity, pricetype)]
    l1, l2, l3, cur, r6 = (float(st[c]) for c in ('lag_1', 'lag_2', 'lag_3', 'usdprice', 'rolling_6m'))
    # rolling_6m is the mean of lag_1..lag_6, so lag_4..lag_6 average (6*r6 - l1 - l2 - l3) / 3
    older = max((6 * r6 - l1 - l2 - l3) / 3, 1e-6)
    prices = [older, older, older, l3, l2, l1, cur]          # oldest ... latest
    now_month = pd.Timestamp.now().month
    out = []
    for step in range(1, months + 1):
        month = (now_month + step - 1) % 12 + 1
        p = predict_from_lags(
            key, province, commodity, pricetype, month,
            prices[-1], prices[-2], prices[-3],
            float(np.mean(prices[-3:])), float(np.mean(prices[-6:])),
        )
        prices.append(p)
        out.append(p)
    return out


def predict_price(commodity, province, pricetype, version=DEFAULT_VERSION, key=None, months=1):
    if (commodity, pricetype) not in reference_stats.index:
        return None, None, None
    pred = forecast_prices(key or reg_key_for(version), commodity, province, pricetype, months)[-1]
    current = reference_stats.loc[(commodity, pricetype), 'usdprice']
    return pred, current, ((pred - current) / current) * 100


_BLIND_CACHE = {}
PROVINCE_NOTE_NL = "\n\nℹ️ Same forecast for every province: the stored inputs are national averages."
PROVINCE_NOTE = ("ℹ️ This model is fed the same inputs for every province, "
                 "so it gives the same forecast everywhere.")


def is_province_blind(key):
    """True if the model returns the same forecast in every province for the stored inputs.
    The stored stats are national averages, so a model that barely uses province_enc can't
    tell provinces apart, and any 'cheapest province' ranking would be arbitrary."""
    if key not in _BLIND_CACHE:
        sample = [c for c in COMMODITIES if (c, 'Retail') in reference_stats.index][::6]
        blind = True
        for c in sample:
            vals = {round(forecast_prices(key, c, p, 'Retail', 1)[0], 8) for p in PROVINCES}
            if len(vals) > 1:
                blind = False
                break
        _BLIND_CACHE[key] = blind
    return _BLIND_CACHE[key]


def predict_margin(commodity, province, version=DEFAULT_VERSION):
    key_r, key_w = (commodity, 'Retail'), (commodity, 'Wholesale')
    if key_r not in reference_stats.index or key_w not in reference_stats.index:
        return None
    retail = float(reference_stats.loc[key_r, 'usdprice'])
    wholesale = float(reference_stats.loc[key_w, 'usdprice'])
    margin_pct = (retail - wholesale) / wholesale * 100

    # NEW uses real recent margins as the lag features. history holds the latest
    # observed margins for the commodity (across markets), not a single market's series.
    history = MARGIN_HISTORY.get(commodity) if version == "new" else None
    if history is not None and len(history) >= 2:
        lag_1, lag_2 = float(history[-1]), float(history[-2])
        rolling = float(np.mean(history[-3:]))
        used_history = True
    else:  # OLD behaviour: all three lag features are today's average margin
        lag_1 = lag_2 = rolling = margin_pct
        used_history = False

    month = pd.Timestamp.now().month
    mf = get_month_features(month)
    X = pd.DataFrame([{
        'commodity_enc': commodity_to_code[commodity],
        'market_enc': province_to_code[province],
        'month': month,
        'year': mf['year'],
        'is_wet_season': mf['is_wet_season'],
        'wholesale_price': wholesale,
        'lag_margin': lag_1,
        'lag_margin_2': lag_2,
        'rolling_margin': rolling,
    }])[MARGIN_FEATURES]

    clf_entry = CLF_MODELS[clf_key_for(version)]
    p_high = float(clf_entry["model"].predict_proba(X)[0][1])
    return {
        "retail": retail, "wholesale": wholesale, "margin_pct": margin_pct,
        "p_high": p_high, "high": p_high > 0.5,
        "model_name": clf_entry["name"], "used_history": used_history,
    }


def anomaly_table(version=DEFAULT_VERSION, province=None, ptypes=("Retail",)):
    """Score every (province, commodity, pricetype) row with the Isolation Forest.

    OLD: the stats are national averages, so every province gets identical rows.
    NEW: uses province_stats.pkl, so each province is scored on its own prices
         (retail only). Falls back to OLD if that file isn't present.
    """
    month = pd.Timestamp.now().month
    if version == "new" and PROVINCE_STATS is not None:
        src = PROVINCE_STATS.reset_index().rename(columns={"admin1": "province"})
        src["pricetype"] = "Retail"
        if province:
            src = src[src["province"] == province]
        src = src[src["pricetype"].isin(ptypes)]
    else:
        base = reference_stats.reset_index()
        base = base[base["pricetype"].isin(ptypes)]
        provs = [province] if province else PROVINCES
        src = pd.concat([base.assign(province=p) for p in provs], ignore_index=True)

    src = src.copy()
    if src.empty:
        return src.assign(flag=False, deviation=0.0)

    X = pd.DataFrame({
        'usdprice': src['usdprice'],
        'lag_1': src['lag_1'],
        'price_momentum': src['lag_1'] - src['lag_2'],
        'rolling_3m': src['rolling_3m'],
        'is_wet_season': get_month_features(month)['is_wet_season'],
        'month': month,
    })[ANOM_FEATURES]
    src["flag"] = iso.predict(scaler.transform(X)) == -1
    src["deviation"] = (src['usdprice'] - src['rolling_3m']) / src['rolling_3m'] * 100
    return src


# =============================================================================
# Text builders for the old-vs-new demo
# =============================================================================
def describe_versions():
    old = (
        "🕰️ *OLD* (before the improvement)\n"
        f"• Price: `{REG_MODELS['v1']['name']}` (predicts the price directly)\n"
        f"• Margin: `{CLF_MODELS['v1']['name']}`\n"
        "• Anomaly: national averages (same result for every province)\n"
    )

    price = (f"`{REG_MODELS[NEW_REG_KEY]['name']}` (predicts % change)" if NEW_PRICE
             else "⚠️ improved model not found, same as old")
    if NEW_MARGIN_MODEL and NEW_MARGIN_INPUTS:
        margin = f"`{CLF_MODELS[NEW_CLF_KEY]['name']}` + recent margin history"
    elif NEW_MARGIN_MODEL:
        margin = f"`{CLF_MODELS[NEW_CLF_KEY]['name']}`"
    elif NEW_MARGIN_INPUTS:
        margin = "original classifier + recent margin history"
    else:
        margin = "⚠️ not found, same as old"
    anomaly = "province-level averages" if NEW_ANOMALY else "⚠️ `province_stats.pkl` not found, same as old"

    new = (
        "🆕 *NEW* (improved)\n"
        f"• Price: {price}\n"
        f"• Margin: {margin}\n"
        f"• Anomaly: {anomaly}"
    )
    return old + "\n" + new


def model_panel(version):
    text = (
        f"🧠 *Active version: {VERSION_LABEL[version]}*\n\n"
        f"{describe_versions()}\n\n"
        "Try `/compare_models Rice, Phnom Penh` to see both side by side, "
        "or `/benchmark` for the test scores."
    )
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(("✅ " if version == "old" else "") + "🕰️ Old", callback_data="model:old"),
        InlineKeyboardButton(("✅ " if version == "new" else "") + "🆕 New", callback_data="model:new"),
    ]])
    return text, keyboard


def benchmark_text():
    out = "📊 *Model benchmark*\n\n"
    if BENCH:
        base = BENCH.get("baseline_mae")
        out += "_Source: benchmark.json. Scored on the newest 20% of months, which no model saw in training._\n\n"
        out += "🔮 *Price forecast* (avg $ error, lower is better)\n```\n"
        out += f"{'Model':<26}{'MAE $':>8}{'vs naive':>9}\n"
        if base:
            out += f"{'Naive (last month)':<26}{base:>8.4f}{'-':>9}\n"
        reg = sorted(BENCH.get("regression", {}).items(), key=lambda kv: kv[1]["MAE"])
        for key, m in reg:
            mark = "* " if key == BENCH.get("default_reg") else "  "
            out += f"{fit(mark + m['name'], 26):<26}{m['MAE']:>8.4f}{m['vs_baseline']:>+8.1f}%\n"
        out += "```\n"
        out += "📊 *Margin classifier* (higher is better)\n```\n"
        out += f"{'Model':<26}{'Acc':>7}{'AUC':>7}\n"
        clf = sorted(BENCH.get("classification", {}).items(), key=lambda kv: -kv[1]["Accuracy"])
        for key, m in clf:
            mark = "* " if key == BENCH.get("default_clf") else "  "
            out += f"{fit(mark + m['name'], 26):<26}{m['Accuracy'] * 100:>6.1f}%{m['AUC']:>7.3f}\n"
        out += "* = model the NEW version uses\n```\n\n"
    else:
        fb = FALLBACK_BENCH
        out += ("_Source: notebook sections 10.1 and 10.3, date-based split. "
                "Add models/benchmark.json for the full model table._\n\n")
        out += "🔮 *Price forecast* (avg $ error, lower is better)\n```\n"
        out += f"{'Model':<24}{'MAE $':>8}{'vs naive':>9}\n"
        out += f"{'Naive (last month)':<24}{fb['baseline_mae']:>8.4f}{'-':>9}\n"
        for name, mae, vs in fb["price"]:
            out += f"{name:<24}{mae:>8.4f}{vs:>+8.1f}%\n"
        out += "```\n"
        out += "📊 *Margin classifier* (higher is better)\n```\n"
        out += f"{'Model':<24}{'Acc':>7}{'AUC':>7}\n"
        out += f"{'Same as last month':<24}{fb['margin_baseline_acc'] * 100:>6.1f}%{'-':>7}\n"
        for name, acc, auc in fb["margin"]:
            out += f"{name:<24}{acc * 100:>6.1f}%{auc:>7.3f}\n"
        out += "```\n\n"

    out += (
        "🚨 *Anomaly detector* (no labels, so no accuracy)\n"
        "At 5% it flags 6.2% of records in the 2020-2022 shock years vs 4.3% in normal years. "
        "At 1% the ratio is 4.7x.\n\n"
        "_Original XGBoost scored 0.1154 MAE on the notebook's first split, but that split tested "
        "on commodities it had never seen. These date-based scores are the fair ones._"
    )
    return out


# =============================================================================
# Guided mode: send a command with no details and the bot asks with buttons
# =============================================================================
def split_args(text):
    """Split on commas, but not commas inside parentheses, so full commodity names
    such as 'Rice (mixed, low quality)' survive."""
    return [p.strip() for p in re.split(r",(?![^()]*\))", text) if p.strip()]


async def need_commodity(update, text):
    found = fuzzy_match(text, COMMODITIES)
    if found:
        return found
    msg = f"❌ Commodity `{text}` not found."
    suggestions = fuzzy_suggestions(text, COMMODITIES)
    if suggestions:
        msg += "\n\n💡 Did you mean:\n" + "\n".join([f"• {s}" for s in suggestions])
    await update.effective_message.reply_text(msg, parse_mode='Markdown')
    return None


async def need_province(update, text):
    found = fuzzy_match(text, PROVINCES)
    if found:
        return found
    msg = f"❌ Province `{text}` not found."
    suggestions = fuzzy_suggestions(text, PROVINCES)
    if suggestions:
        msg += "\n\n💡 Did you mean:\n" + "\n".join([f"• {s}" for s in suggestions])
    await update.effective_message.reply_text(msg, parse_mode='Markdown')
    return None


# flow name -> (title, ordered steps)
WIZ_FLOWS = {
    "predict":        ("🔮 Price prediction",  ["commodity", "province", "pricetype", "horizon"]),
    "anomaly":        ("🚨 Anomaly check",     ["commodity_all", "province"]),
    "margin":         ("📊 Margin prediction", ["commodity", "province"]),
    "summary":        ("📋 Province summary",  ["province", "horizon"]),
    "compare":        ("⚖️ Province comparison", ["commodity", "province", "province2", "horizon"]),
    "chart":          ("📈 Price chart",       ["commodity", "province", "horizon_chart"]),
    "alert":          ("🔔 Price alert",       ["commodity", "province", "threshold"]),
    "compare_models": ("🧪 Old vs New",        ["commodity", "province", "pricetype", "horizon"]),
    "top":            ("🏆 Rankings",          ["topmode", "commodity", "horizon"]),
}
STEP_KEY = {"commodity_all": "commodity", "horizon_chart": "horizon"}   # which value a step fills in
COMMODITY_PAGE, PROVINCE_PAGE = 12, 14
WIZ_IDS = itertools.count(1)
TOP_MODES = ("anomalies", "expensive", "cheapest")


def wiz_steps(w):
    steps = list(WIZ_FLOWS[w["flow"]][1])
    if w["flow"] == "top" and w["vals"].get("topmode") == "anomalies":
        steps = [s for s in steps if s not in ("commodity", "horizon")]   # anomaly ranking is current, all commodities
    return steps


def wiz_current(w):
    for s in wiz_steps(w):
        if STEP_KEY.get(s, s) not in w["vals"]:
            return s
    return None


def wiz_options(w, step):
    if step in ("commodity", "commodity_all"):
        return COMMODITIES
    if step == "province2":
        return [p for p in PROVINCES if p != w["vals"].get("province")]
    return PROVINCES


def wiz_threshold_options(commodity):
    key = (commodity, "Retail")
    if key not in reference_stats.index:
        return []
    avg = float(reference_stats.loc[key, "usdprice"])
    return [(f"+{p}%  (${avg * (1 + p / 100):.4f})", f"{avg * (1 + p / 100):.4f}") for p in (5, 10, 20)]


def wiz_summary(w):
    v = w["vals"]
    lines = [f"*{WIZ_FLOWS[w['flow']][0]}*"]
    if "topmode" in v:
        lines.append(f"🏆 `{v['topmode']}`")
    if "commodity" in v:
        lines.append(f"📦 `{'All commodities' if v['commodity'] == '*' else v['commodity']}`")
    if "province" in v:
        lines.append(f"📍 `{v['province']}`")
    if "province2" in v:
        lines.append(f"📍 `{v['province2']}`")
    if "pricetype" in v:
        lines.append(f"🏪 `{v['pricetype']}`")
    if "horizon" in v:
        n = int(v["horizon"])
        lines.append(f"⏳ `{'next month' if n == 1 else f'{n} months ahead'}`")
    if "threshold" in v:
        lines.append(f"💰 `${float(v['threshold']):.4f}`")
    return "\n".join(lines)


PROMPTS = {
    "commodity": "Pick a commodity:", "commodity_all": "Pick a commodity:",
    "province": "Pick a province:", "province2": "Pick the second province:",
    "pricetype": "Retail or wholesale prices?", "topmode": "Which ranking?",
    "threshold": "Alert me when the predicted price reaches:",
    "horizon": "How far ahead should I predict?", "horizon_chart": "How many months should the chart forecast?",
}


def wiz_render(w):
    step = wiz_current(w)
    wid = w["id"]
    btn = InlineKeyboardButton
    rows = []
    prompt = PROMPTS[step]

    if step in ("commodity", "commodity_all", "province", "province2"):
        items = wiz_options(w, step)
        is_c = step.startswith("commodity")
        size = COMMODITY_PAGE if is_c else PROVINCE_PAGE
        pages = max(1, -(-len(items) // size))
        page = min(w["page"], pages - 1)
        if step == "commodity_all":
            rows.append([btn("🌐 All commodities", callback_data=f"w:{wid}:v:*")])
        row = []
        for off, name in enumerate(items[page * size:(page + 1) * size]):
            row.append(btn(fit(name, 36 if is_c else 18), callback_data=f"w:{wid}:p:{page * size + off}"))
            if len(row) == (1 if is_c else 2):
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        if pages > 1:
            nav = []
            if page > 0:
                nav.append(btn("⬅️ Prev", callback_data=f"w:{wid}:g:{page - 1}"))
            nav.append(btn(f"{page + 1}/{pages}", callback_data=f"w:{wid}:n:0"))
            if page < pages - 1:
                nav.append(btn("Next ➡️", callback_data=f"w:{wid}:g:{page + 1}"))
            rows.append(nav)
            prompt += f" (page {page + 1}/{pages})"
    elif step == "pricetype":
        rows.append([btn("🏪 Retail", callback_data=f"w:{wid}:v:Retail"),
                     btn("📦 Wholesale", callback_data=f"w:{wid}:v:Wholesale")])
    elif step in ("horizon", "horizon_chart"):
        opts = HORIZONS if step == "horizon" else CHART_HORIZONS
        btns = [btn("1 month" if n == 1 else f"{n} months", callback_data=f"w:{wid}:v:{n}") for n in opts]
        rows.append(btns[:3])
        if btns[3:]:
            rows.append(btns[3:])
    elif step == "topmode":
        rows.append([btn("🚨 Most anomalies", callback_data=f"w:{wid}:v:anomalies")])
        rows.append([btn("💸 Most expensive", callback_data=f"w:{wid}:v:expensive"),
                     btn("💰 Cheapest", callback_data=f"w:{wid}:v:cheapest")])
    elif step == "threshold":
        for label, value in wiz_threshold_options(w["vals"]["commodity"]):
            rows.append([btn(label, callback_data=f"w:{wid}:v:{value}")])

    ctl = []
    if w["vals"]:
        ctl.append(btn("⬅️ Back", callback_data=f"w:{wid}:b:0"))
    ctl.append(btn("✖️ Cancel", callback_data=f"w:{wid}:x:0"))
    rows.append(ctl)
    return f"{wiz_summary(w)}\n\n{prompt}", InlineKeyboardMarkup(rows)


async def start_wizard(update, context, flow):
    w = {"id": next(WIZ_IDS), "flow": flow, "vals": {}, "page": 0}
    context.user_data["wiz"] = w
    text, keyboard = wiz_render(w)
    await update.effective_message.reply_text(text, reply_markup=keyboard, parse_mode='Markdown')


async def wiz_run(update, context, flow, v):
    months = int(v["horizon"]) if "horizon" in v else 1
    if flow == "predict":
        await run_predict(update, context, v["commodity"], v["province"], v["pricetype"], months)
    elif flow == "anomaly":
        await run_anomaly(update, context, v["province"], None if v["commodity"] == "*" else v["commodity"])
    elif flow == "margin":
        await run_margin(update, context, v["commodity"], v["province"])
    elif flow == "summary":
        await run_summary(update, context, v["province"], months)
    elif flow == "compare":
        await run_compare(update, context, v["commodity"], v["province"], v["province2"], months)
    elif flow == "chart":
        await run_chart(update, context, v["commodity"], v["province"], months)
    elif flow == "alert":
        await run_alert(update, context, v["commodity"], v["province"], float(v["threshold"]))
    elif flow == "compare_models":
        await run_compare_models(update, context, v["commodity"], v["province"], v["pricetype"], months)
    elif flow == "top":
        await run_top(update, context, v["topmode"], v.get("commodity"), months)


async def wizard_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    parts = query.data.split(":", 3)                      # w : <wizard id> : <action> : <argument>
    w = context.user_data.get("wiz")
    if not w or str(w["id"]) != parts[1]:
        await query.answer("This menu has expired. Send the command again.", show_alert=True)
        return
    await query.answer()
    act, arg = parts[2], (parts[3] if len(parts) > 3 else "")

    async def show(text, keyboard=None):
        try:
            await query.edit_message_text(text, reply_markup=keyboard, parse_mode='Markdown')
        except BadRequest:
            pass   # same content as before, nothing to update

    if act == "x":
        context.user_data.pop("wiz", None)
        await show(f"{wiz_summary(w)}\n\n✖️ Cancelled")
        return
    if act == "n":
        return

    step = wiz_current(w)
    if act == "g":
        w["page"] = int(arg)
    elif act == "b":
        for s in reversed(wiz_steps(w)):
            if STEP_KEY.get(s, s) in w["vals"]:
                del w["vals"][STEP_KEY.get(s, s)]
                break
        w["page"] = 0
    elif act == "p" and step in ("commodity", "commodity_all", "province", "province2"):
        items = wiz_options(w, step)
        idx = int(arg)
        if 0 <= idx < len(items):
            w["vals"][STEP_KEY.get(step, step)] = items[idx]
            w["page"] = 0
    elif act == "v":
        ok = ((step == "pricetype" and arg in ("Retail", "Wholesale"))
              or (step == "topmode" and arg in TOP_MODES)
              or (step == "commodity_all" and arg == "*")
              or (step == "horizon" and arg.isdigit() and int(arg) in HORIZONS)
              or (step == "horizon_chart" and arg.isdigit() and int(arg) in CHART_HORIZONS))
        if step == "threshold":
            try:
                ok = float(arg) > 0
            except ValueError:
                ok = False
        if ok:
            w["vals"][STEP_KEY.get(step, step)] = arg
            w["page"] = 0

    if wiz_current(w) is None:
        context.user_data.pop("wiz", None)
        await show(f"{wiz_summary(w)}\n\n_Fetching result…_")
        await wiz_run(update, context, w["flow"], w["vals"])
    else:
        text, keyboard = wiz_render(w)
        await show(text, keyboard)


# --- main menu ---------------------------------------------------------------
MENU_ITEMS = [
    ("🔮 Predict price", "predict"), ("🚨 Anomalies", "anomaly"),
    ("📊 Margin", "margin"), ("📋 Province summary", "summary"),
    ("⚖️ Compare provinces", "compare"), ("🏆 Rankings", "top"),
    ("📈 Price chart", "chart"), ("🔔 Price alert", "alert"),
    ("🧪 Old vs New", "compare_models"), ("🧠 Model version", "model"),
    ("📊 Benchmark", "benchmark"),
]


def menu_keyboard():
    rows, row = [], []
    for label, key in MENU_ITEMS:
        row.append(InlineKeyboardButton(label, callback_data=f"m:{key}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("What would you like to do?", reply_markup=menu_keyboard())


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    key = query.data[2:]
    if key == "model":
        await model_cmd(update, context)
    elif key == "benchmark":
        await benchmark(update, context)
    elif key in WIZ_FLOWS:
        await start_wizard(update, context, key)


BOT_COMMANDS = [
    ("menu", "Open the button menu"), ("predict", "Predict next month's price"),
    ("anomaly", "Check for unusual prices"), ("margin", "Retail markup prediction"),
    ("summary", "Price overview for a province"), ("compare", "Compare two provinces"),
    ("top", "Rankings across provinces"), ("chart", "Price trend chart"),
    ("alert", "Set a price alert"), ("model", "Switch old / new models"),
    ("compare_models", "Old vs new, side by side"), ("benchmark", "Model test scores"),
    ("help", "Usage guide"),
]


async def post_init(application):
    """Fill Telegram's '/' command menu so users can tap a command instead of typing it."""
    try:
        await application.bot.set_my_commands([BotCommand(c, d) for c, d in BOT_COMMANDS])
    except Exception as e:
        log.warning("Could not set the command menu: %s", e)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "🇰🇭 *Cambodia Food Price Prediction Bot*\n\n"
        "👇 *Tap a button below*, or send any command on its own "
        "(like `/anomaly`) and I'll ask you step by step.\n\n"
        "Or type everything in one go:\n\n"
        "🔮 `/predict <commodity>, <province>, <Retail|Wholesale>[, <months>]`\n"
        "🚨 `/anomaly <province>`\n"
        "📊 `/margin <commodity>, <province>`\n"
        "📋 `/summary <province>`\n"
        "⚖️ `/compare <commodity>, <province1>, <province2>`\n"
        "🏆 `/top <anomalies|expensive|cheapest>`\n"
        "📈 `/chart <commodity>, <province>`\n"
        "🔔 `/alert <commodity>, <province>, <price>`\n"
        "🛒 `/commodities` — list all commodities\n"
        "🗺️ `/provinces` — list all provinces\n"
        "❓ `/help` — detailed usage guide\n\n"
        "*Old vs improved models:*\n"
        "📱 `/menu` — button menu\n"
        "🧠 `/model` — switch between old and new\n"
        "🧪 `/compare_models <commodity>, <province>` — side by side\n"
        "📊 `/benchmark` — test scores\n\n"
        "*Examples:*\n"
        "`/predict Rice, Phnom Penh, Retail`\n"
        "`/compare Rice, Phnom Penh, Battambang`\n"
        "`/top anomalies`\n"
        "`/chart Rice, Phnom Penh`\n"
        "`/alert Rice, Phnom Penh, 0.60`",
        reply_markup=menu_keyboard(),
        parse_mode='Markdown'
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "❓ *Help & Usage Guide*\n\n"
        "🔮 */predict* `<commodity>, <province>, <Retail|Wholesale>`\n"
        "Predicts the price 1 to 12 months ahead (default: next month)\n"
        "Example: `/predict Rice, Phnom Penh, Retail, 3`\n\n"
        "🚨 */anomaly* `<province>`\n"
        "Detects unusual prices using Isolation Forest\n"
        "Example: `/anomaly Kampong Cham`\n\n"
        "📊 */margin* `<commodity>, <province>`\n"
        "Predicts whether retail markup is above the median (Gradient Boosting)\n"
        "Example: `/margin Oil, Battambang`\n\n"
        "📋 */summary* `<province>`\n"
        "Full price overview for a province\n"
        "Example: `/summary Phnom Penh`\n\n"
        "⚖️ */compare* `<commodity>, <province1>, <province2>`\n"
        "Side by side price comparison between two provinces\n"
        "Example: `/compare Rice, Phnom Penh, Battambang`\n\n"
        "🏆 */top* `<anomalies|expensive|cheapest>`\n"
        "Rankings across all provinces\n"
        "Example: `/top anomalies`\n\n"
        "📈 */chart* `<commodity>, <province>`\n"
        "Price trend chart with old and new forecasts\n"
        "Example: `/chart Rice, Phnom Penh`\n\n"
        "🔔 */alert* `<commodity>, <province>, <price>`\n"
        "Set price alert threshold\n"
        "Example: `/alert Rice, Phnom Penh, 0.60`\n\n"
        "🧠 */model* `[old|new]`\n"
        "Choose which model version every command uses\n\n"
        "🧪 `/compare_models <commodity>, <province>[, <Retail|Wholesale>]`\n"
        "Old vs new prediction, margin and anomaly check side by side\n\n"
        "📊 */benchmark*\n"
        "Test-set scores for each model\n\n"
        "💡 *Tips:*\n"
        "• Send a command with nothing after it and I'll ask with buttons\n"
        "• Add a number to look further ahead: `/chart Rice, Kep, 6`\n"
        "• Forecasts beyond 1 month are rougher (the model feeds itself)\n"
        "• `/anomaly Rice, Phnom Penh` checks one commodity\n"
        "• `/top expensive Pork` ranks any commodity\n"
        "• Commas separate inputs\n"
        "• Partial names work: 'Rice' → 'Rice (mixed, low quality)'\n"
        "• Bot suggests closest match if not found",
        parse_mode='Markdown'
    )


async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        choice = context.args[0].lower()
        if choice not in ("old", "new"):
            await update.effective_message.reply_text("❌ Usage: `/model old` or `/model new`", parse_mode='Markdown')
            return
        if choice == "new" and not HAS_NEW:
            await update.effective_message.reply_text(
                f"⚠️ No improved models found in `{MODELS_DIR}/` yet, so the bot is still on OLD.",
                parse_mode='Markdown'
            )
            return
        context.user_data["version"] = choice
    text, keyboard = model_panel(get_version(context))
    await update.effective_message.reply_text(text, reply_markup=keyboard, parse_mode='Markdown')


async def benchmark(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(benchmark_text(), parse_mode='Markdown')


async def compare_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "compare_models")
    parts = split_args(" ".join(context.args))
    if not 2 <= len(parts) <= 4:
        await update.message.reply_text(
            "❌ Usage: `/compare_models <commodity>, <province>[, <Retail|Wholesale>][, <months>]`\n"
            "Example: `/compare_models Rice, Phnom Penh, Retail, 3`",
            parse_mode='Markdown'
        )
        return
    pricetype, months = 'Retail', 1
    for extra in parts[2:]:                 # the optional bits can come in either order
        if extra.capitalize() in ('Retail', 'Wholesale'):
            pricetype = extra.capitalize()
        elif parse_months(extra) is not None:
            months = parse_months(extra)
        else:
            await update.message.reply_text(
                f"❌ `{extra}` is neither `Retail`/`Wholesale` nor a number of months (1 to 12)",
                parse_mode='Markdown')
            return
    commodity = await need_commodity(update, parts[0])
    province = await need_province(update, parts[1]) if commodity else None
    if commodity and province:
        await run_compare_models(update, context, commodity, province, pricetype, months)


async def run_compare_models(update: Update, context: ContextTypes.DEFAULT_TYPE,
                             commodity, province, pricetype, months=1):
    try:
        stats_key = (commodity, pricetype)
        if stats_key not in reference_stats.index:
            await update.effective_message.reply_text(f"❌ No data for `{commodity}` ({pricetype})", parse_mode='Markdown')
            return

        stats = reference_stats.loc[stats_key]
        current = stats['usdprice']

        def row(label, price):
            chg = (price - current) / current * 100
            return f"{fit(label, 25):<25} ${price:>7.4f}{chg:>+7.1f}%\n"

        # --- price ---
        text = (
            f"🧪 *Old vs New*\n"
            f"📦 `{commodity}`\n📍 `{province}` · 🏪 `{pricetype}`\n\n"
            f"🔮 *Price for {horizon_label(months)}*\n(current avg `${current:.4f}`)\n```\n"
            + row("Naive (no change)", current)
        )
        for key, entry in REG_MODELS.items():
            tag = "OLD  " if key == "v1" else ("NEW  " if key == NEW_REG_KEY else "     ")
            price = forecast_prices(key, commodity, province, pricetype, months)[-1]
            text += row(tag + entry["name"], price)
        text += "```\n"
        if not NEW_PRICE:
            text += "⚠️ Improved price model not found, so there is nothing to compare yet.\n"
        elif months == 1:
            bench_reg = (BENCH or {}).get("regression", {})
            if BENCH and NEW_REG_KEY in bench_reg:
                text += f"_Test MAE of the NEW model: ${bench_reg[NEW_REG_KEY]['MAE']:.4f}, naive ${BENCH['baseline_mae']:.4f}_\n"
            else:
                fb = FALLBACK_BENCH
                text += (f"_Test MAE: OLD ${fb['price'][0][1]:.4f}, NEW ${fb['price'][1][1]:.4f}, "
                         f"naive ${fb['baseline_mae']:.4f}_\n")

        if months > 1:
            text += f"{LONG_RANGE_NOTE}\n"

        # --- margin ---
        old_m = predict_margin(commodity, province, "old")
        new_m = predict_margin(commodity, province, "new")
        if old_m:
            text += (
                f"\n📊 *Markup vs median ({MEDIAN_MARGIN}%)*\n"
                f"Current margin: `{old_m['margin_pct']:.1f}%`\n```\n"
            )
            for tag, m in (("OLD", old_m), ("NEW", new_m)):
                label = "HIGH" if m["high"] else "LOW "
                text += f"{tag}  {label}  P(high)={m['p_high'] * 100:5.1f}%\n"
            text += "```\n"
            if old_m["high"] != new_m["high"]:
                text += "⚠️ The two versions disagree on this one.\n"
            elif not (NEW_MARGIN_MODEL or NEW_MARGIN_INPUTS):
                text += "⚠️ No improved margin model or history found, so NEW matches OLD.\n"

        # --- anomaly (retail only: the NEW stats are retail) ---
        if pricetype == 'Retail':
            text += "\n🚨 *Anomaly check* (retail)\n```\n"
            for tag, ver in (("OLD", "old"), ("NEW", "new")):
                tbl = anomaly_table(ver, province)
                r = tbl[tbl["commodity"] == commodity]
                if r.empty:
                    text += f"{tag}  no data\n"
                else:
                    r = r.iloc[0]
                    state = "ANOMALY" if r["flag"] else "normal "
                    text += f"{tag}  {state}  {r['deviation']:+.1f}% vs 3-mo avg\n"
            text += "```\n"
            if not NEW_ANOMALY:
                text += "⚠️ `province_stats.pkl` not found, so NEW matches OLD.\n"

        await update.effective_message.reply_text(text, parse_mode='Markdown')

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def commodities(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # inline keyboard for commodities
    keyboard = []
    row = []
    for i, c in enumerate(COMMODITIES):
        short = c.split('(')[0].strip()[:20]
        row.append(InlineKeyboardButton(short, callback_data=f"commodity:{c}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text("🛒 *Select a commodity:*", reply_markup=reply_markup, parse_mode='Markdown')


async def provinces(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for i, p in enumerate(PROVINCES):
        row.append(InlineKeyboardButton(p, callback_data=f"province:{p}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text("🗺️ *Select a province:*", reply_markup=reply_markup, parse_mode='Markdown')


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("model:"):
        choice = data.replace("model:", "")
        if choice == "new" and not HAS_NEW:
            await query.answer("No improved models found yet", show_alert=True)
            return
        context.user_data["version"] = choice
        text, keyboard = model_panel(choice)
        try:
            await query.edit_message_text(text, reply_markup=keyboard, parse_mode='Markdown')
        except BadRequest:
            pass  # tapped the version that was already active, nothing to change

    elif data.startswith("commodity:"):
        commodity = data.replace("commodity:", "")
        key_retail = (commodity, 'Retail')
        if key_retail in reference_stats.index:
            stats = reference_stats.loc[key_retail]
            await query.edit_message_text(
                f"📦 *{commodity}*\n\n"
                f"💵 Avg retail price: `${stats['usdprice']:.4f}`\n"
                f"📈 3-month avg: `${stats['rolling_3m']:.4f}`\n"
                f"📊 6-month avg: `${stats['rolling_6m']:.4f}`\n\n"
                f"Use `/predict {commodity.split('(')[0].strip()}, <province>, Retail` for forecast",
                parse_mode='Markdown'
            )

    elif data.startswith("province:"):
        province = data.replace("province:", "")
        await query.edit_message_text(
            f"📍 *{province}*\n\n"
            f"Use `/summary {province}` for full price overview\n"
            f"Use `/anomaly {province}` for anomaly detection",
            parse_mode='Markdown'
        )


# =============================================================================
# Weekly broadcast
# Triggered by the GitHub Action (broadcast_once.py) or manually with /broadcast.
# It is NOT scheduled inside the bot anymore, so restarts never spam the channel.
# =============================================================================
async def weekly_broadcast(app):
    month = pd.Timestamp.now().month
    month_feats = get_month_features(month)
    version = DEFAULT_VERSION

    # top 5 commodities by price change
    predictions = []
    for commodity in COMMODITIES:
        key = (commodity, 'Retail')
        if key not in reference_stats.index:
            continue
        pred, current, change = predict_price(commodity, 'Phnom Penh', 'Retail', version)
        if pred:
            color, status = price_trend_emoji(change)
            predictions.append((commodity, current, pred, change, color))

    predictions.sort(key=lambda x: abs(x[3]), reverse=True)

    # anomaly count (distinct commodities flagged)
    tbl = anomaly_table(version)
    anomaly_count = tbl.loc[tbl["flag"], "commodity"].nunique()

    text = (
        f"📢 *Weekly Cambodia Food Price Report*\n"
        f"📅 {pd.Timestamp.now().strftime('%A, %d %B %Y')}\n"
        f"🌡️ Season: {'🌧️ Wet' if month_feats['is_wet_season'] else '☀️ Dry'}\n\n"
        f"*Top Price Movements this week:*\n"
    )

    for commodity, current, pred, change, color in predictions[:5]:
        short = commodity.split('(')[0].strip()
        text += f"{color} `{short}` `${pred:.3f}` `{change:+.1f}%`\n"

    text += (
        f"\n🚨 Anomalies detected: `{anomaly_count}` commodities\n\n"
        f"_Powered by ML models trained on WFP Cambodia 2018-2026 data_\n"
        f"_Use @FoodPricePredictionBot bot for detailed predictions_"
    )

    await app.bot.send_message(
        chat_id=CHANNEL_ID,
        text=text,
        parse_mode='Markdown'
    )
    print("Weekly broadcast sent ✅")


# =============================================================================
# Commands. Each has two halves:
#   <name>()      reads typed arguments (or starts the guided buttons if there are none)
#   run_<name>()  does the work with exact, already-validated names
# =============================================================================
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "predict")
    parts = split_args(" ".join(context.args))
    months = 1
    if len(parts) == 4:
        months = parse_months(parts[3])
        if months is None:
            await update.message.reply_text("❌ Months must be a number from 1 to 12", parse_mode='Markdown')
            return
        parts = parts[:3]
    if len(parts) != 3:
        await update.message.reply_text(
            "❌ Usage: `/predict <commodity>, <province>, <Retail|Wholesale>[, <months>]`\n"
            "Example: `/predict Rice, Phnom Penh, Retail, 3`\n"
            "Or send `/predict` on its own and pick with buttons.",
            parse_mode='Markdown'
        )
        return
    commodity_input, province_input, pricetype = parts
    pricetype = pricetype.capitalize()
    if pricetype not in ['Retail', 'Wholesale']:
        await update.message.reply_text("❌ Price type must be `Retail` or `Wholesale`", parse_mode='Markdown')
        return
    commodity = await need_commodity(update, commodity_input)
    province = await need_province(update, province_input) if commodity else None
    if commodity and province:
        await run_predict(update, context, commodity, province, pricetype, months)


async def run_predict(update: Update, context: ContextTypes.DEFAULT_TYPE,
                      commodity, province, pricetype, months=1):
    try:
        version = get_version(context)
        if (commodity, pricetype) not in reference_stats.index:
            await update.effective_message.reply_text(f"❌ No data for `{commodity}` ({pricetype})", parse_mode='Markdown')
            return

        path = forecast_prices(reg_key_for(version), commodity, province, pricetype, months)
        stats = reference_stats.loc[(commodity, pricetype)]
        current, pred = stats['usdprice'], path[-1]
        change = (pred - current) / current * 100
        color, status = price_trend_emoji(change)
        month_feats = get_month_features(pd.Timestamp.now().month)

        # check alerts
        alert_key = f"{update.effective_user.id}:{commodity}:{province}"
        alert_msg = ""
        if alert_key in alerts:
            threshold = alerts[alert_key]
            if pred >= threshold:
                alert_msg = f"\n\n⚠️ *ALERT TRIGGERED!* Predicted price `${pred:.4f}` exceeds your alert threshold of `${threshold:.4f}`"

        month_by_month = ""
        if months > 1:
            now = pd.Timestamp.now()
            rows = "".join(f"{(now + pd.DateOffset(months=i)).strftime('%b %Y')}  ${p:.4f}\n"
                           for i, p in enumerate(path, 1))
            month_by_month = f"🗓️ *Month by month*\n```\n{rows}```\n{LONG_RANGE_NOTE}\n\n"

        await update.effective_message.reply_text(
            f"🔮 *Price Prediction*\n\n"
            f"📦 Commodity: `{commodity}`\n"
            f"📍 Province: `{province}`\n"
            f"🏪 Type: `{pricetype}`\n\n"
            f"💵 Current avg: `${current:.4f}`\n"
            f"🎯 Predicted for {horizon_label(months)}: `${pred:.4f}`\n"
            f"{color} Change: `{change:+.2f}%` — {status}\n\n"
            f"{month_by_month}"
            f"📊 *Reasoning:*\n"
            f"• 3-month avg: `${stats['rolling_3m']:.4f}`\n"
            f"• 6-month avg: `${stats['rolling_6m']:.4f}`\n"
            f"• Season now: `{'🌧️ Wet' if month_feats['is_wet_season'] else '☀️ Dry'}`\n"
            f"• Momentum: `{'↑ Rising' if stats['lag_1'] > stats['lag_2'] else '↓ Falling'}`\n\n"
            f"{model_footer(version)}"
            f"{PROVINCE_NOTE_NL if is_province_blind(reg_key_for(version)) else ''}"
            f"{alert_msg}",
            parse_mode='Markdown'
        )

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def anomaly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "anomaly")
    parts = split_args(" ".join(context.args))
    if len(parts) == 2:
        commodity = await need_commodity(update, parts[0])
        if not commodity:
            return
        province = await need_province(update, parts[1])
    elif len(parts) == 1:
        commodity = None
        province = await need_province(update, parts[0])
    else:
        await update.effective_message.reply_text(
            "❌ Usage: `/anomaly <province>` or `/anomaly <commodity>, <province>`\n"
            "Example: `/anomaly Rice, Phnom Penh`",
            parse_mode='Markdown'
        )
        return
    if province:
        await run_anomaly(update, context, province, commodity)


async def run_anomaly(update: Update, context: ContextTypes.DEFAULT_TYPE, province, commodity=None):
    try:
        version = get_version(context)
        tbl = anomaly_table(version, province, ("Retail", "Wholesale"))

        if commodity:   # one commodity: show the verdict for each price type
            rows = tbl[tbl["commodity"] == commodity]
            text = f"🚨 *Anomaly check*\n📦 `{commodity}`\n📍 `{province}`\n\n"
            if rows.empty:
                text += "No data for this commodity.\n"
            for _, r in rows.iterrows():
                state = "🚨 ANOMALY" if r["flag"] else "✅ Normal"
                text += f"• {r['pricetype']}: {state} (`{r['deviation']:+.1f}%` vs 3-month avg)\n"
        else:
            flagged = tbl[tbl["flag"]].sort_values("deviation", key=abs, ascending=False)
            if not flagged.empty:
                text = f"🚨 *Anomalies in {province}:*\n\n"
                for _, r in flagged.head(15).iterrows():
                    direction = "📈" if r["deviation"] > 0 else "📉"
                    text += f"{direction} `{r['commodity']}` ({r['pricetype']}) — `{r['deviation']:+.1f}%` from avg\n"
                if len(flagged) > 15:
                    text += f"\n...and {len(flagged) - 15} more"
                text += f"\n\n*Total: {len(flagged)} anomalies*"
            else:
                text = f"✅ *No anomalies in {province}*\nPrices appear stable."

        stats_note = ("province-level averages (retail only)" if version == "new" and NEW_ANOMALY
                      else "national averages")
        text += f"\n\n🧠 Version: {VERSION_LABEL[version]} · uses {stats_note}"
        await update.effective_message.reply_text(text, parse_mode='Markdown')

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def margin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "margin")
    parts = split_args(" ".join(context.args))
    if len(parts) != 2:
        await update.effective_message.reply_text(
            "❌ Usage: `/margin <commodity>, <province>`\n"
            "Example: `/margin Rice, Phnom Penh`",
            parse_mode='Markdown'
        )
        return
    commodity = await need_commodity(update, parts[0])
    province = await need_province(update, parts[1]) if commodity else None
    if commodity and province:
        await run_margin(update, context, commodity, province)


async def run_margin(update: Update, context: ContextTypes.DEFAULT_TYPE, commodity, province):
    try:
        version = get_version(context)
        res = predict_margin(commodity, province, version)
        if res is None:
            await update.effective_message.reply_text(f"❌ No retail/wholesale data for `{commodity}`", parse_mode='Markdown')
            return

        month = pd.Timestamp.now().month
        label = "🔴 HIGH margin" if res["high"] else "🟢 LOW margin"
        margin_pct = res["margin_pct"]

        await update.effective_message.reply_text(
            f"📊 *Margin Prediction*\n\n"
            f"📦 Commodity: `{commodity}`\n"
            f"📍 Province: `{province}`\n\n"
            f"💵 Avg wholesale: `${res['wholesale']:.4f}`\n"
            f"💰 Avg retail: `${res['retail']:.4f}`\n"
            f"📈 Current margin: `{margin_pct:.1f}%`\n"
            f"📊 Median margin: `{MEDIAN_MARGIN}%`\n\n"
            f"🎯 Prediction: {label}\n"
            f"🔢 P(high margin): `{res['p_high'] * 100:.1f}%`\n\n"
            f"📊 *Reasoning:*\n"
            f"• Margin `{'above' if margin_pct > MEDIAN_MARGIN else 'below'}` median ({MEDIAN_MARGIN}%)\n"
            f"• {'⚠️ Consumers paying significantly more than wholesale' if res['high'] else '✅ Market fairly priced'}\n"
            f"• Season: `{'🌧️ Wet — supply may be affected' if month in [5, 6, 7, 8, 9, 10] else '☀️ Dry — stable supply'}`\n"
            f"• Inputs: `{'recent margin history' if res['used_history'] else 'current avg margin only'}`\n\n"
            f"🧠 Model: `{res['model_name']}` ({VERSION_LABEL[version]})",
            parse_mode='Markdown'
        )

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "summary")
    parts = split_args(" ".join(context.args))
    months = 1
    if len(parts) == 2:
        months = parse_months(parts[1])
        if months is None:
            await update.message.reply_text("❌ Months must be a number from 1 to 12", parse_mode='Markdown')
            return
    province = await need_province(update, parts[0])
    if province:
        await run_summary(update, context, province, months)


async def run_summary(update: Update, context: ContextTypes.DEFAULT_TYPE, province, months=1):
    try:
        version = get_version(context)
        month = pd.Timestamp.now().month
        month_feats = get_month_features(month)
        predictions = []

        for commodity in COMMODITIES:
            key = (commodity, 'Retail')
            if key not in reference_stats.index:
                continue
            pred, current, change = predict_price(commodity, province, 'Retail', version, months=months)
            if pred is None:
                continue
            color, _ = price_trend_emoji(change)
            predictions.append((commodity, current, pred, change, color))

        anomaly_count = int(anomaly_table(version, province)["flag"].sum())
        predictions.sort(key=lambda x: abs(x[3]), reverse=True)

        text = f"📋 *Market Summary — {province}*\n"
        text += f"📅 {pd.Timestamp.now().strftime('%B %Y')}\n"
        text += f"🌡️ Season: {'🌧️ Wet' if month_feats['is_wet_season'] else '☀️ Dry'}\n"
        text += f"⏳ Forecast: {horizon_label(months)}\n\n"
        text += f"*Top Price Movements (Retail):*\n"

        for commodity, current, pred, change, color in predictions[:10]:
            short_name = commodity.split('(')[0].strip()[:15]
            text += f"{color} `{short_name:<15}` `${pred:.3f}` `{change:+.1f}%`\n"

        text += f"\n🚨 Anomalies now: `{anomaly_count}` commodities\n"
        text += f"📦 Tracked: `{len(predictions)}` commodities\n\n"
        if months > 1:
            text += f"{LONG_RANGE_NOTE}\n\n"
        if is_province_blind(reg_key_for(version)):
            text += f"{PROVINCE_NOTE}\n\n"
        text += "_Use /predict for detailed forecast_\n"
        text += "_Use /anomaly for anomaly details_\n\n"
        text += model_footer(version)

        await update.effective_message.reply_text(text, parse_mode='Markdown')

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "compare")
    parts = split_args(" ".join(context.args))
    months = 1
    if len(parts) == 4:
        months = parse_months(parts[3])
        if months is None:
            await update.message.reply_text("❌ Months must be a number from 1 to 12", parse_mode='Markdown')
            return
        parts = parts[:3]
    if len(parts) != 3:
        await update.message.reply_text(
            "❌ Usage: `/compare <commodity>, <province1>, <province2>[, <months>]`\n"
            "Example: `/compare Rice, Phnom Penh, Battambang, 3`",
            parse_mode='Markdown'
        )
        return
    commodity = await need_commodity(update, parts[0])
    province1 = await need_province(update, parts[1]) if commodity else None
    province2 = await need_province(update, parts[2]) if province1 else None
    if commodity and province1 and province2:
        await run_compare(update, context, commodity, province1, province2, months)


async def run_compare(update: Update, context: ContextTypes.DEFAULT_TYPE,
                      commodity, province1, province2, months=1):
    try:
        version = get_version(context)
        pred1, current1, change1 = predict_price(commodity, province1, 'Retail', version, months=months)
        pred2, current2, change2 = predict_price(commodity, province2, 'Retail', version, months=months)

        if pred1 is None or pred2 is None:
            await update.effective_message.reply_text(f"❌ No data available for comparison", parse_mode='Markdown')
            return

        when = horizon_label(months)
        color1, status1 = price_trend_emoji(change1)
        color2, status2 = price_trend_emoji(change2)
        diff = abs(pred1 - pred2)
        if is_province_blind(reg_key_for(version)) or diff < 5e-5:
            verdict = f"🟰 Same predicted price in both provinces.\n{PROVINCE_NOTE}\n\n"
        else:
            cheaper = province1 if pred1 < pred2 else province2
            verdict = (f"🏆 Cheaper in {when}: `{cheaper}`\n"
                       f"💰 Price difference: `${diff:.4f}`\n\n")
        long_range = f"{LONG_RANGE_NOTE}\n\n" if months > 1 else ""

        await update.effective_message.reply_text(
            f"⚖️ *Price Comparison*\n"
            f"📦 Commodity: `{commodity}`\n"
            f"⏳ Forecast: {when}\n\n"
            f"📍 *{province1}*\n"
            f"💵 Current: `${current1:.4f}` → Predicted: `${pred1:.4f}`\n"
            f"{color1} Change: `{change1:+.2f}%` — {status1}\n\n"
            f"📍 *{province2}*\n"
            f"💵 Current: `${current2:.4f}` → Predicted: `${pred2:.4f}`\n"
            f"{color2} Change: `{change2:+.2f}%` — {status2}\n\n"
            f"{verdict}"
            f"{long_range}"
            f"{model_footer(version)}",
            parse_mode='Markdown'
        )

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def top(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "top")
    mode = context.args[0].lower()
    if mode not in TOP_MODES:
        await update.message.reply_text("❌ Mode must be `anomalies`, `expensive`, or `cheapest`", parse_mode='Markdown')
        return
    commodity, months = None, 1
    if mode != "anomalies" and len(context.args) > 1:   # /top expensive Pork, 6   or   /top cheapest 3
        parts = split_args(" ".join(context.args[1:]))
        if parse_months(parts[0]) is not None and len(parts) == 1:
            months = parse_months(parts[0])               # only a month count was given
        else:
            commodity = await need_commodity(update, parts[0])
            if not commodity:
                return
            if len(parts) > 1:
                months = parse_months(parts[1])
                if months is None:
                    await update.message.reply_text("❌ Months must be a number from 1 to 12", parse_mode='Markdown')
                    return
    await run_top(update, context, mode, commodity, months)


async def run_top(update: Update, context: ContextTypes.DEFAULT_TYPE, mode, commodity=None, months=1):
    try:
        version = get_version(context)

        if mode == 'anomalies':
            tbl = anomaly_table(version)
            counts = (tbl[tbl["flag"]].groupby("province").size()
                      .reindex(PROVINCES, fill_value=0).sort_values(ascending=False))
            text = "🏆 *Top Provinces by Anomalies:*\n\n"
            medals = ['🥇', '🥈', '🥉']
            for i, (province, count) in enumerate(counts.head(10).items()):
                medal = medals[i] if i < 3 else f"{i+1}."
                text += f"{medal} `{province}` — `{count}` anomalies\n"
            if version == "old" or not NEW_ANOMALY:
                text += "\nℹ️ Every province gets the same count here because the stats are national averages."
                if HAS_NEW and NEW_ANOMALY:
                    text += " Try `/model new`."
            text += f"\n\n🧠 Version: {VERSION_LABEL[version]}"

        else:   # expensive / cheapest
            commodity = commodity or fuzzy_match('Rice', COMMODITIES)
            when = horizon_label(months)
            province_prices = {}
            for province in PROVINCES:
                pred, current, change = predict_price(commodity, province, 'Retail', version, months=months)
                if pred:
                    province_prices[province] = pred
            if not province_prices:
                await update.effective_message.reply_text(f"❌ No retail data for `{commodity}`", parse_mode='Markdown')
                return

            if is_province_blind(reg_key_for(version)):
                await update.effective_message.reply_text(
                    f"🟰 *No ranking possible*\n📦 `{commodity}`\n⏳ {when}\n\n"
                    f"All {len(province_prices)} provinces get the same predicted price "
                    f"(`${next(iter(province_prices.values())):.4f}`), so any ranking would be arbitrary.\n"
                    f"{PROVINCE_NOTE}\n\n{model_footer(version)}",
                    parse_mode='Markdown'
                )
                return

            reverse = mode == 'expensive'
            sorted_prices = sorted(province_prices.items(), key=lambda x: x[1], reverse=reverse)
            emoji = "💸" if reverse else "💰"
            text = (f"{emoji} *{'Most Expensive' if reverse else 'Cheapest'} by Province*\n"
                    f"📦 `{commodity}`\n⏳ {when}\n\n")
            medals = ['🥇', '🥈', '🥉']
            for i, (province, price) in enumerate(sorted_prices[:10]):
                medal = medals[i] if i < 3 else f"{i+1}."
                text += f"{medal} `{province}` — `${price:.4f}`\n"
            if months > 1:
                text += f"\n{LONG_RANGE_NOTE}"
            text += f"\n\n{model_footer(version)}"

        await update.effective_message.reply_text(text, parse_mode='Markdown')

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "chart")
    parts = split_args(" ".join(context.args))
    months = 3
    if len(parts) == 3:
        months = parse_months(parts[2])
        if months is None:
            await update.message.reply_text("❌ Months must be a number from 1 to 12", parse_mode='Markdown')
            return
        parts = parts[:2]
    if len(parts) != 2:
        await update.message.reply_text(
            "❌ Usage: `/chart <commodity>, <province>[, <months>]`\n"
            "Example: `/chart Rice, Phnom Penh, 6`",
            parse_mode='Markdown'
        )
        return
    commodity = await need_commodity(update, parts[0])
    province = await need_province(update, parts[1]) if commodity else None
    if commodity and province:
        await run_chart(update, context, commodity, province, months)


async def run_chart(update: Update, context: ContextTypes.DEFAULT_TYPE, commodity, province, months=3):
    try:
        if (commodity, 'Retail') not in reference_stats.index:
            await update.effective_message.reply_text(f"❌ No data for `{commodity}`", parse_mode='Markdown')
            return

        stats = reference_stats.loc[(commodity, 'Retail')]
        history = [stats['lag_3'], stats['lag_2'], stats['lag_1'], stats['usdprice']]
        now = pd.Timestamp.now()
        labels = ['3mo ago', '2mo ago', 'Last mo', 'Current'] + [
            (now + pd.DateOffset(months=i)).strftime('%b %y') for i in range(1, months + 1)]

        # forecast with the old model, and with the new one when it's different
        old_fc = forecast_prices("v1", commodity, province, 'Retail', months)
        new_fc = forecast_prices(NEW_REG_KEY, commodity, province, 'Retail', months) if NEW_PRICE else None
        xs = range(3, 4 + months)

        fig, ax = plt.subplots(figsize=(10 if months <= 3 else 12, 5))
        ax.plot(range(4), history, marker='o', color='steelblue', linewidth=2, label='Historical')
        ax.plot(xs, [history[-1]] + old_fc, marker='o', color='gray',
                linestyle='--', linewidth=2, label='Forecast — OLD')
        if new_fc is not None:
            ax.plot(xs, [history[-1]] + new_fc, marker='o', color='darkorange',
                    linewidth=2, label='Forecast — NEW')
        ax.axvline(x=3, color='gray', linestyle=':', alpha=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45 if months > 3 else 0, ha='right' if months > 3 else 'center')
        ax.set_title(f'{commodity}\n{province} — Price Trend & {months}-Month Forecast', fontsize=13)
        ax.set_ylabel('USD Price')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)

        end = (now + pd.DateOffset(months=months)).strftime('%B %Y')
        caption = (f"📈 *{commodity} — {province}*\n"
                   f"Blue = Historical | Gray = Old forecast"
                   f"{' | Orange = New forecast' if new_fc is not None else ''}\n"
                   f"Current: `${stats['usdprice']:.4f}` → {end}: old `${old_fc[-1]:.4f}`")
        if new_fc is not None:
            caption += f", new `${new_fc[-1]:.4f}`"
        if months > 1:
            caption += f"\n{LONG_RANGE_NOTE}"

        await update.effective_message.reply_photo(photo=buf, caption=caption, parse_mode='Markdown')

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        return await start_wizard(update, context, "alert")
    parts = split_args(" ".join(context.args))
    if len(parts) != 3:
        await update.effective_message.reply_text(
            "❌ Usage: `/alert <commodity>, <province>, <price threshold>`\n"
            "Example: `/alert Rice, Phnom Penh, 0.60`",
            parse_mode='Markdown'
        )
        return
    try:
        threshold = float(parts[2])
    except ValueError:
        await update.effective_message.reply_text("❌ Price threshold must be a number", parse_mode='Markdown')
        return
    commodity = await need_commodity(update, parts[0])
    province = await need_province(update, parts[1]) if commodity else None
    if commodity and province:
        await run_alert(update, context, commodity, province, threshold)


async def run_alert(update: Update, context: ContextTypes.DEFAULT_TYPE, commodity, province, threshold):
    try:
        alert_key = f"{update.effective_user.id}:{commodity}:{province}"
        alerts[alert_key] = threshold

        # check immediately
        pred, current, change = predict_price(commodity, province, 'Retail', get_version(context))
        if pred is None:
            await update.effective_message.reply_text(f"❌ No retail data for `{commodity}`", parse_mode='Markdown')
            return
        status = ""
        if pred >= threshold:
            status = f"\n\n⚠️ *Alert already triggered!* Predicted price `${pred:.4f}` is above your threshold."

        await update.effective_message.reply_text(
            f"🔔 *Alert Set!*\n\n"
            f"📦 Commodity: `{commodity}`\n"
            f"📍 Province: `{province}`\n"
            f"💰 Alert when price ≥ `${threshold:.4f}`\n"
            f"💵 Current avg: `${current:.4f}`\n\n"
            f"_Alert will trigger on your next `/predict` command_"
            f"{status}",
            parse_mode='Markdown'
        )

    except Exception as e:
        await update.effective_message.reply_text(f"❌ Error: {str(e)}")


async def broadcast_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.effective_message.reply_text("❌ Unauthorized")
        return
    await weekly_broadcast(context.application)
    await update.effective_message.reply_text("✅ Broadcast sent!")


# --- Main ---
def main():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .build()
    )

    # add all
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("commodities", commodities))
    app.add_handler(CommandHandler("provinces", provinces))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("anomaly", anomaly))
    app.add_handler(CommandHandler("margin", margin))
    app.add_handler(CommandHandler("summary", summary))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(CommandHandler("top", top))
    app.add_handler(CommandHandler("chart", chart))
    app.add_handler(CommandHandler("alert", alert))
    app.add_handler(CommandHandler("broadcast", broadcast_now))
    # old vs improved demo
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("compare_models", compare_models))
    app.add_handler(CommandHandler("benchmark", benchmark))
    app.add_handler(CommandHandler("menu", menu_cmd))
    # guided buttons first, so their callbacks never reach the generic handler
    app.add_handler(CallbackQueryHandler(wizard_callback, pattern=r"^w:"))
    app.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^m:"))
    app.add_handler(CallbackQueryHandler(button_handler))

    log.info("Models loaded: price=%s | margin=%s | new anomaly stats=%s | default version=%s",
             {k: v["name"] for k, v in REG_MODELS.items()},
             {k: v["name"] for k, v in CLF_MODELS.items()},
             NEW_ANOMALY, DEFAULT_VERSION)

    base_url = os.getenv("RENDER_EXTERNAL_URL")
    if base_url:
        # Render: webhook mode (Render sets RENDER_EXTERNAL_URL and PORT itself)
        print("Bot running 🚀 (webhook)")
        app.run_webhook(
            listen="0.0.0.0",
            port=int(os.getenv("PORT", 10000)),
            url_path=BOT_TOKEN,
            webhook_url=f"{base_url}/{BOT_TOKEN}",
            drop_pending_updates=True,
        )
    else:
        # local: plain polling
        print("Bot running 🚀 (polling)")
        app.run_polling()


if __name__ == "__main__":
    main()