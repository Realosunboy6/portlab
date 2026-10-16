"""Ticker universe: 11 GICS sectors (~400 curated stocks), sector ETFs,
and 22 extended ETF categories — carried over from PULL-DATA2026 /
Portfolio_Optimization_COLAB — plus asset-class proxies with ETF inception
dates.
"""

# ── Universe definition ───────────────────────────────────────
SECTORS = {
    "Communication Services": [
        "LYV","GOOG","META","DIS","NFLX","T","VZ","TMUS","CABO","NWSA",
        "TTWO","EA","NXST","FOX","WBD","ROKU","CHTR","OMC","ZG","SIRI",
        "SPOT","TCEHY","CCOI","PINS","SNAP","BIDU","NTES","SE","BILI","CMCSA",
        "ZM","VOD","RCI","LUMN","AMX","SKM","NTTYY",
    ],
    "Consumer Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","TGT","BKNG",
        "GM","F","ROST","MAR","LVS","ORLY","YUM","EBAY","AZO","DHI",
        "LEN","RCL","HLT","CMG","NVR","DRI","ULTA","DG","KMX","BBY",
        "ETSY","HAS","GRMN","MGM","WHR","TPR","RL","GPC","WSM","BURL",
        "POOL","CZR","WYNN",
    ],
    "Consumer Staples": [
        "PG","KO","PEP","WMT","PM","MO","COST","MDLZ","UL","CL",
        "KHC","KMB","GIS","STZ","ADM","MNST","SYY","EL","TAP","CPB",
        "CAG","HSY","TSN","MKC","CHD","KDP","CLX","LW","HRL","SJM",
        "BG","TSCO","SFM","KR","SAM","POST","INGR","FLO",
    ],
    "Energy": [
        "XOM","CVX","BP","EQT","COP","EOG","PBR","KMI","PSX","VLO",
        "MPC","WMB","SU","OXY","TRGP","CTRA","APA","DVN","SLB","LNG",
        "OVV","AR","RRC","HAL","BKR","SM","MTDR","PAA","MUR","NOG","OKE",
    ],
    "Financials": [
        "JPM","BAC","WFC","C","AXP","BK","BLK","SCHW","USB","PNC",
        "TFC","COF","STT","CB","AIG","MET","PRU","ALL","TRV","AFL",
        "PFG","AMP","CME","ICE","SPGI","MCO","NDAQ","V","MA","PYPL",
        "AON","MS","GS","SYF","ALLY","KEY","RF","FITB","HBAN",
    ],
    "Health Care": [
        "JNJ","UNH","PFE","ABBV","LLY","MRK","TMO","ABT","DHR","BMY",
        "AMGN","MDT","GILD","CVS","ISRG","CNC","ELV","SYK","ZTS","REGN",
        "VRTX","BSX","BDX","ILMN","BIIB","EW","IQV","MCK","CI","NVO",
        "CAH","HOLX","RMD","HCA","LH","DGX","IDXX","DVA","UHS","HUM",
    ],
    "Industrials": [
        "HON","UNP","UPS","RTX","CAT","BA","DE","GE","MMM","LMT",
        "FDX","NOC","GD","EMR","ITW","ETN","CSX","WM","CMI","WAB",
        "LHX","GWW","PH","ROP","TT","DAL","AAL","UAL","SWK","FAST",
        "RSG","CTAS","XYL","IR","PNR","ROL","SNA","URI","JBHT","DOV","PWR",
    ],
    "Information Technology": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CSCO","ACN","ADBE","TXN","IBM",
        "CRM","AMD","INTC","QCOM","AMAT","INTU","ADI","MU","LRCX","NOW",
        "PAYX","ADP","CTSH","CDNS","SNPS","KLAC","MCHP","NXPI","ADSK","FTNT",
        "PANW","HPQ","KEYS","GLW","STX","WDAY","SWKS","MPWR","ON","APH",
    ],
    "Materials": [
        "LIN","SHW","APD","ECL","NEM","FCX","DD","VMC","MLM","LYB",
        "PPG","DOW","NUE","IFF","ALB","EMN","MOS","FMC","AVY","CF",
        "AA","IP","PKG","BALL","HUN","RS","STLD","CLF","RPM","OLN",
        "OC","SQM",
    ],
    "Real Estate": [
        "PLD","AMT","EQIX","SPG","PSA","CCI","WELL","O","DLR","EXR",
        "AVB","EQR","ESS","VTR","CUBE","MAA","INVH","UDR","SUI","CPT",
        "WY","KIM","FRT","REG","IRM","HST","VNO","NNN","BRX","STAG",
        "OHI","LTC","BXP","REXR","ADC","ARE",
    ],
    "Utilities": [
        "NEE","SO","DUK","D","AEP","EXC","SRE","XEL","PEG","ED",
        "WEC","ES","EIX","DTE","PPL","FE","AEE","CMS","ATO","NI",
        "CNP","EVRG","LNT","PNW","AES","NRG","VST","BKH","IDA","UGI",
        "AWK","HE",
    ],
}

SECTOR_ETFS = {
    "Communication Services" : "XLC",
    "Consumer Discretionary" : "XLY",
    "Consumer Staples"        : "XLP",
    "Energy"                  : "XLE",
    "Financials"              : "XLF",
    "Health Care"             : "XLV",
    "Industrials"             : "XLI",
    "Information Technology"  : "XLK",
    "Materials"               : "XLB",
    "Real Estate"             : "XLRE",
    "Utilities"               : "XLU",
}

# ── Extended ETF universe (22 categories) ─────────────────────
ETF_CATEGORIES = {
    "Broad Market": [
        "SPY","QQQ","IWM","VTI","IVW","IVE","VUG","VTV","VOO","DIA",
        "IWF","IWD","ITOT","SCHB","SPTM","MDY","IJH",
    ],
    "International": [
        "EFA","EEM","EWJ","EWZ","MCHI","INDA","VEU","IEFA","IEMG","VWO",
        "EWG","EWU","EWC","EWA","EWH","EWS","EWT","EWY","EWP","EWI",
        "EWQ","EWN","EWL","FXI","KWEB","GXC","THD",
    ],
    "Fixed Income": [
        "AGG","BND","TLT","IEF","LQD","HYG","TIP","EMB","SHY","SHV",
        "VCIT","VCSH","VGSH","VGIT","VGLT","MUB","BNDX","IGIB","USIG",
        "SJNK","JNK","SCHZ","GOVT","SPTL","MBB",
    ],
    "Commodities": [
        "GLD","SLV","USO","UNG","DBC","GDX","GDXJ","PPLT","PALL","CPER",
        "WEAT","CORN","SOYB","DBA","DJP","GSG","COMT","BCI","FTGC",
        "PDBC","USCI","RJI","SGG","NIB","JO","REMX","URA",
    ],
    "Tech Innovation": [
        "BOTZ","CIBR","SMH","SKYY","BITO","ARKK","SOXX","HACK","ROBO",
        "FINX","IPAY","CLOU","WCLD","AIQ","GNOM","DRIV","PRNT","IZRL",
        "BUG","IGV","FDN","XSW",
    ],
    "Healthcare ETF": [
        "ARKG","XBI","IBB","IHI","MSOS","IHE","IHF",
    ],
    "Clean Energy": [
        "ICLN","TAN","FAN","LIT","QCLN","PBW","ACES","PBD","CNRG",
        "SMOG","CTEC",
    ],
    "ESG": [
        "ESGU","SUSA","EAGG","CRBN","DSI","SNPE",
    ],
    "ARK Innovation": [
        "ARKW","ARKQ","ARKF","ARKX","ARKG",
    ],
    "Industries ETF": [
        "KRE","XHB","XOP","XES","ITA","REM","JETS","XME","XRT","PAVE",
        "WOOD","PHO","CGW","NFRA","IFRA","GRID","BLOK","BETZ","DFEN",
        "IGE","OIH","AMLP",
    ],
    "Growth Factors": [
        "VOT","VBK","IUSG","VONG","IWO",
    ],
    "Value Factors": [
        "VOE","VBR","IUSV","VONV","IWN",
    ],
    "Dividend Factors": [
        "VYM","VIG","NOBL","DGRO","SDY","SCHD","HDV","DVY","SPHD","SPYD",
    ],
    "Quality LowVol Momentum": [
        "QUAL","USMV","MTUM","SPLV","LRGF","VLUE","SIZE","FVAL","FDMO",
        "JMOM","JQUA","JVAL","JPSE","JPIN","JPHF",
    ],
    "Leveraged 2x": [
        "SSO","QLD","UCO","UGL","UBT","ROM","UYG","URE","UWM","SAA",
        "MVV","UCC",
    ],
    "Leveraged 3x": [
        "UPRO","TQQQ","SOXL","FAS","TMF","TECL","CURE","TNA","LABU",
        "NAIL","DPST","RETL","UDOW","UMDD","URTY","DRN","EDC","YINN",
        "WANT","MIDU","HIBL","FNGU","BULZ",
    ],
    "Inverse 1x": [
        "SH","PSQ","DOG","RWM","HDGE","TAIL","MYY","SBB","SEF","EUM",
        "EFZ",
    ],
    "Inverse 2x": [
        "SDS","QID","TBT","SCO","SRS","TWM","DXD","SKF","EPV","EEV",
        "BZQ","SSG","SDD","SJB","SMN","MZZ","SDP","ZSL","DUST",
    ],
    "Inverse 3x": [
        "SPXU","SQQQ","FAZ","SOXS","TZA","SDOW","SRTY","SMDD","TMV",
        "LABD","TECS","YANG","WEBS","DRV","EDZ",
    ],
    "Currency": [
        "UUP","FXE","FXY","FXB","CYB","FXA","FXC","FXF","USDU","CEW",
        "DBV","UDN","FXS","FXSG","CNY","EUO",
    ],
    "Volatility": [
        "VXX","UVXY","SVXY","VIXM","VIXY","VXZ","SVOL","LSVX",
    ],
    "RE Sub-Sectors": [
        "REZ","INDS","VNQI","USRT","ICF","RWR","BBRE","FREL","REET",
        "SCHH",
    ],
}

# ── Build flat list + sector map ──────────────────────────────
ALL_TICKERS = []
SECTOR_MAP  = {}

# 1. GICS sector stocks
for sec, tickers in SECTORS.items():
    for t in tickers:
        if t not in SECTOR_MAP:
            ALL_TICKERS.append(t)
            SECTOR_MAP[t] = sec

# 2. Sector ETFs (one per GICS sector)
for sec, etf in SECTOR_ETFS.items():
    if etf not in SECTOR_MAP:
        ALL_TICKERS.append(etf)
        SECTOR_MAP[etf] = sec + " ETF"

# 3. Extended ETF categories
for cat, tickers in ETF_CATEGORIES.items():
    for t in tickers:
        if t not in SECTOR_MAP:
            ALL_TICKERS.append(t)
            SECTOR_MAP[t] = cat

ALL_TICKERS = list(dict.fromkeys(ALL_TICKERS))


# ── Asset-class proxies (asset classes -> investable ETFs) ─────────────
# value = (ticker, approximate inception year of the ETF history)
ASSET_CLASS_PROXIES = {
    "US Stock Market":            ("VTI", 2001),
    "US Large Cap":               ("SPY", 1993),
    "US Large Cap Growth":        ("VUG", 2004),
    "US Large Cap Value":         ("VTV", 2004),
    "US Mid Cap":                 ("MDY", 1995),
    "US Small Cap":               ("IWM", 2000),
    "US Small Cap Value":         ("VBR", 2004),
    "Intl Developed Markets":     ("EFA", 2001),
    "Emerging Markets":           ("EEM", 2003),
    "Global ex-US":               ("VEU", 2007),
    "US REIT":                    ("VNQ", 2004),
    "Total US Bond Market":       ("BND", 2007),
    "Intermediate Treasury":      ("IEF", 2002),
    "Long Term Treasury":         ("TLT", 2002),
    "Short Term Treasury":        ("SHY", 2002),
    "TIPS":                       ("TIP", 2003),
    "Corporate Bonds":            ("LQD", 2002),
    "High Yield Bonds":           ("HYG", 2007),
    "Intl Bonds":                 ("BNDX", 2013),
    "Gold":                       ("GLD", 2004),
    "Commodities":                ("DBC", 2006),
    "Cash (T-Bills)":             ("BIL", 2007),
    "Bitcoin (spot proxy)":       ("BTC-USD", 2014),
    "Ethereum (spot proxy)":      ("ETH-USD", 2017),
}


def sector_tickers(sector: str | None = None) -> list[str]:
    """Stocks for one GICS sector, or all sectors when None."""
    if sector is None:
        return [t for lst in SECTORS.values() for t in lst]
    return list(SECTORS[sector])


def all_stocks() -> list[str]:
    return sector_tickers(None)


def etf_universe(category: str | None = None) -> list[str]:
    if category is None:
        return [t for lst in ETF_CATEGORIES.values() for t in lst]
    return list(ETF_CATEGORIES[category])


def asset_class_tickers(names: list[str] | None = None) -> dict[str, str]:
    """Map asset-class names to proxy tickers (all classes when None)."""
    src = ASSET_CLASS_PROXIES
    names = names or list(src)
    return {n: src[n][0] for n in names}
