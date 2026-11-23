"""Company directory for the financial corpora: ticker -> display name, sector, aliases.

One map for everything that turns a ticker or a company name into something else:
the clarifier (names in questions, "similar companies" by sector, banks), the
manifest reader (fills ``sector`` for manifests written before the field existed)
and both indexers (``sector`` on every chunk).

Sectors are the 13 of the sector corpus plan (docs/annual_report_corpus_plan.html),
which come from SEC industry codes plus overrides. That is why Amazon is Retail,
Tesla is Consumer goods and Visa is Financial services.
tests/test_financial_clarifier.py checks that every plan company has the plan's sector here.

Aliases are lower-case phrases matched on word boundaries. Leave out words that are
ordinary English or finance terms on their own ("progressive", "southern", "target",
"principal", "permian"), and use the longer name instead.

Standard library only.
"""
from __future__ import annotations

BANKING, INSURANCE, FINANCIAL = "Banking", "Insurance", "Financial services"
OIL_GAS, UTILITIES, HEALTHCARE = "Oil and gas", "Energy and utilities", "Healthcare"
TECH, RETAIL, TELECOM = "Technology and IT", "Retail", "Telecom and media"
INDUSTRIALS, CONSUMER, MATERIALS, REAL_ESTATE = "Industrials", "Consumer goods", "Materials and chemicals", "Real estate"
SECTOR_NAMES = (BANKING, INSURANCE, FINANCIAL, OIL_GAS, UTILITIES, HEALTHCARE, TECH, RETAIL, TELECOM,
                INDUSTRIALS, CONSUMER, MATERIALS, REAL_ESTATE)

COMPANIES: dict[str, tuple[str, str, list[str]]] = {
    # ---------------------------------------------------------------- Technology and IT
    "AAPL": ("Apple", TECH, ["apple"]),
    "MSFT": ("Microsoft", TECH, ["microsoft"]),
    "NVDA": ("NVIDIA", TECH, ["nvidia"]),
    "GOOGL": ("Alphabet", TECH, ["alphabet", "google"]),
    "META": ("Meta", TECH, ["meta platforms", "meta"]),
    "AVGO": ("Broadcom", TECH, ["broadcom"]),
    "ORCL": ("Oracle", TECH, ["oracle"]),
    "CRM": ("Salesforce", TECH, ["salesforce"]),
    "AMD": ("AMD", TECH, ["advanced micro devices"]),
    "ADBE": ("Adobe", TECH, ["adobe"]),
    "CSCO": ("Cisco", TECH, ["cisco"]),
    "INTC": ("Intel", TECH, ["intel"]),
    "IBM": ("IBM", TECH, ["ibm"]),
    "QCOM": ("Qualcomm", TECH, ["qualcomm"]),
    "TXN": ("Texas Instruments", TECH, ["texas instruments"]),
    "ACN": ("Accenture", TECH, ["accenture"]),
    "ADI": ("Analog Devices", TECH, ["analog devices"]),
    "ADP": ("ADP", TECH, ["automatic data processing"]),
    "AMAT": ("Applied Materials", TECH, ["applied materials"]),
    "ANET": ("Arista Networks", TECH, ["arista networks", "arista"]),
    "APH": ("Amphenol", TECH, ["amphenol"]),
    "APP": ("AppLovin", TECH, ["applovin"]),
    "CRWD": ("CrowdStrike", TECH, ["crowdstrike"]),
    "DELL": ("Dell Technologies", TECH, ["dell technologies", "dell"]),
    "GEV": ("GE Vernova", TECH, ["ge vernova", "vernova"]),
    "KLAC": ("KLA", TECH, ["kla corporation", "kla corp", "kla-tencor", "kla"]),
    "LRCX": ("Lam Research", TECH, ["lam research"]),
    "MRVL": ("Marvell Technology", TECH, ["marvell technology", "marvell"]),
    "MU": ("Micron Technology", TECH, ["micron technology", "micron"]),
    "NOW": ("ServiceNow", TECH, ["servicenow"]),
    "PANW": ("Palo Alto Networks", TECH, ["palo alto networks"]),
    "PLTR": ("Palantir", TECH, ["palantir"]),
    "SNDK": ("Sandisk", TECH, ["sandisk"]),
    "STX": ("Seagate Technology", TECH, ["seagate technology", "seagate"]),
    "UBER": ("Uber", TECH, ["uber technologies", "uber"]),
    # ---------------------------------------------------------------- Banking
    "JPM": ("JPMorgan Chase", BANKING, ["jpmorgan", "jp morgan", "chase"]),
    "BAC": ("Bank of America", BANKING, ["bank of america", "bofa"]),
    "WFC": ("Wells Fargo", BANKING, ["wells fargo"]),
    "C": ("Citigroup", BANKING, ["citigroup", "citibank", "citi"]),
    "ALLY": ("Ally Financial", BANKING, ["ally financial", "ally bank"]),
    "BNY": ("BNY", BANKING, ["bank of new york mellon", "bny mellon"]),
    "BOKF": ("BOK Financial", BANKING, ["bok financial"]),
    "BPOP": ("Popular", BANKING, ["popular, inc", "popular inc", "banco popular"]),
    "CBC": ("Central Bancompany", BANKING, ["central bancompany"]),
    "CBSH": ("Commerce Bancshares", BANKING, ["commerce bancshares", "commerce bank"]),
    "CFG": ("Citizens Financial Group", BANKING, ["citizens financial"]),
    "CFR": ("Cullen/Frost Bankers", BANKING, ["cullen/frost", "cullen frost", "frost bank"]),
    "COF": ("Capital One", BANKING, ["capital one"]),
    "COLB": ("Columbia Banking System", BANKING, ["columbia banking"]),
    "EWBC": ("East West Bancorp", BANKING, ["east west bancorp", "east west bank"]),
    "FCNCA": ("First Citizens BancShares", BANKING, ["first citizens"]),
    "FHN": ("First Horizon", BANKING, ["first horizon"]),
    "FITB": ("Fifth Third Bancorp", BANKING, ["fifth third"]),
    "HBAN": ("Huntington Bancshares", BANKING, ["huntington bancshares", "huntington bank"]),
    "KEY": ("KeyCorp", BANKING, ["keycorp", "keybank"]),
    "MTB": ("M&T Bank", BANKING, ["m&t bank", "m&t"]),
    "NTRS": ("Northern Trust", BANKING, ["northern trust"]),
    "ONB": ("Old National Bancorp", BANKING, ["old national"]),
    "PB": ("Prosperity Bancshares", BANKING, ["prosperity bancshares", "prosperity bank"]),
    "RF": ("Regions Financial", BANKING, ["regions financial", "regions bank"]),
    "STT": ("State Street", BANKING, ["state street"]),
    "TFC": ("Truist Financial", BANKING, ["truist"]),
    "USB": ("U.S. Bancorp", BANKING, ["u.s. bancorp", "us bancorp", "u.s. bank"]),
    "WAL": ("Western Alliance Bancorporation", BANKING, ["western alliance"]),
    "WTFC": ("Wintrust Financial", BANKING, ["wintrust"]),
    "ZION": ("Zions Bancorporation", BANKING, ["zions bancorporation", "zions bank", "zions"]),
    # ---------------------------------------------------------------- Financial services
    "V": ("Visa", FINANCIAL, ["visa"]),
    "MA": ("Mastercard", FINANCIAL, ["mastercard"]),
    "AXP": ("American Express", FINANCIAL, ["american express", "amex"]),
    "GS": ("Goldman Sachs", FINANCIAL, ["goldman sachs", "goldman"]),
    "MS": ("Morgan Stanley", FINANCIAL, ["morgan stanley"]),
    "AMP": ("Ameriprise Financial", FINANCIAL, ["ameriprise"]),
    "APO": ("Apollo Global Management", FINANCIAL, ["apollo global"]),
    "ARES": ("Ares Management", FINANCIAL, ["ares management"]),
    "BLK": ("BlackRock", FINANCIAL, ["blackrock"]),
    "CME": ("CME Group", FINANCIAL, ["cme group"]),
    "HOOD": ("Robinhood Markets", FINANCIAL, ["robinhood"]),
    "IBKR": ("Interactive Brokers", FINANCIAL, ["interactive brokers"]),
    "ICE": ("Intercontinental Exchange", FINANCIAL, ["intercontinental exchange"]),
    "MCO": ("Moody's", FINANCIAL, ["moody's", "moodys"]),
    "MSCI": ("MSCI", FINANCIAL, ["msci inc"]),
    "PYPL": ("PayPal", FINANCIAL, ["paypal"]),
    "RKT": ("Rocket Companies", FINANCIAL, ["rocket companies", "rocket mortgage"]),
    "SCHW": ("Charles Schwab", FINANCIAL, ["charles schwab", "schwab"]),
    "SPGI": ("S&P Global", FINANCIAL, ["s&p global"]),
    # ---------------------------------------------------------------- Insurance
    "AFL": ("Aflac", INSURANCE, ["aflac"]),
    "AIG": ("AIG", INSURANCE, ["american international group"]),
    "AJG": ("Arthur J. Gallagher", INSURANCE, ["arthur j. gallagher", "gallagher"]),
    "ALL": ("Allstate", INSURANCE, ["allstate"]),
    "AON": ("Aon", INSURANCE, ["aon"]),
    "BRK-B": ("Berkshire Hathaway", INSURANCE, ["berkshire hathaway", "berkshire"]),
    "BRO": ("Brown & Brown", INSURANCE, ["brown & brown", "brown and brown"]),
    "CB": ("Chubb", INSURANCE, ["chubb"]),
    "CNA": ("CNA Financial", INSURANCE, ["cna financial"]),
    "CRBG": ("Corebridge Financial", INSURANCE, ["corebridge"]),
    "EG": ("Everest Group", INSURANCE, ["everest group", "everest re"]),
    "EQH": ("Equitable Holdings", INSURANCE, ["equitable holdings"]),
    "ERIE": ("Erie Indemnity", INSURANCE, ["erie indemnity", "erie insurance"]),
    "HIG": ("The Hartford", INSURANCE, ["the hartford", "hartford insurance", "hartford financial"]),
    "L": ("Loews", INSURANCE, ["loews"]),
    "MET": ("MetLife", INSURANCE, ["metlife"]),
    "MKL": ("Markel Group", INSURANCE, ["markel"]),
    "MRSH": ("Marsh & McLennan", INSURANCE, ["marsh & mclennan", "marsh mclennan", "marsh and mclennan"]),
    "PFG": ("Principal Financial Group", INSURANCE, ["principal financial"]),
    "PGR": ("Progressive", INSURANCE, ["progressive corporation", "progressive corp", "progressive insurance"]),
    "PRU": ("Prudential Financial", INSURANCE, ["prudential financial", "prudential"]),
    "RGA": ("Reinsurance Group of America", INSURANCE, ["reinsurance group of america"]),
    "RNR": ("RenaissanceRe", INSURANCE, ["renaissancere"]),
    "TRV": ("Travelers", INSURANCE, ["travelers companies", "travelers insurance", "the travelers"]),
    "WRB": ("W. R. Berkley", INSURANCE, ["w. r. berkley", "w.r. berkley", "wr berkley"]),
    "WTW": ("Willis Towers Watson", INSURANCE, ["willis towers watson"]),
    # ---------------------------------------------------------------- Oil and gas
    "XOM": ("Exxon Mobil", OIL_GAS, ["exxon mobil", "exxonmobil", "exxon"]),
    "CVX": ("Chevron", OIL_GAS, ["chevron"]),
    "AR": ("Antero Resources", OIL_GAS, ["antero resources", "antero"]),
    "COP": ("ConocoPhillips", OIL_GAS, ["conocophillips"]),
    "DINO": ("HF Sinclair", OIL_GAS, ["hf sinclair"]),
    "DTM": ("DT Midstream", OIL_GAS, ["dt midstream"]),
    "DVN": ("Devon Energy", OIL_GAS, ["devon energy"]),
    "ENB": ("Enbridge", OIL_GAS, ["enbridge"]),
    "EOG": ("EOG Resources", OIL_GAS, ["eog resources"]),
    "EQT": ("EQT", OIL_GAS, ["eqt corporation", "eqt corp"]),
    "EXE": ("Expand Energy", OIL_GAS, ["expand energy"]),
    "FANG": ("Diamondback Energy", OIL_GAS, ["diamondback energy", "diamondback"]),
    "LNG": ("Cheniere Energy", OIL_GAS, ["cheniere energy", "cheniere"]),
    "MPC": ("Marathon Petroleum", OIL_GAS, ["marathon petroleum"]),
    "OKE": ("ONEOK", OIL_GAS, ["oneok"]),
    "OVV": ("Ovintiv", OIL_GAS, ["ovintiv"]),
    "OXY": ("Occidental Petroleum", OIL_GAS, ["occidental petroleum", "occidental"]),
    "PAA": ("Plains All American Pipeline", OIL_GAS, ["plains all american"]),
    "PR": ("Permian Resources", OIL_GAS, ["permian resources"]),
    "PSX": ("Phillips 66", OIL_GAS, ["phillips 66"]),
    "RRC": ("Range Resources", OIL_GAS, ["range resources"]),
    "SLB": ("SLB", OIL_GAS, ["schlumberger", "slb limited"]),
    "TRGP": ("Targa Resources", OIL_GAS, ["targa resources", "targa"]),
    "VLO": ("Valero Energy", OIL_GAS, ["valero"]),
    "VNOM": ("Viper Energy", OIL_GAS, ["viper energy"]),
    "WMB": ("Williams Companies", OIL_GAS, ["williams companies"]),
    # ---------------------------------------------------------------- Energy and utilities
    "AEE": ("Ameren", UTILITIES, ["ameren"]),
    "AEP": ("American Electric Power", UTILITIES, ["american electric power"]),
    "ATO": ("Atmos Energy", UTILITIES, ["atmos energy"]),
    "AWK": ("American Water Works", UTILITIES, ["american water works", "american water"]),
    "CEG": ("Constellation Energy", UTILITIES, ["constellation energy"]),
    "CMS": ("CMS Energy", UTILITIES, ["cms energy"]),
    "CNP": ("CenterPoint Energy", UTILITIES, ["centerpoint energy", "centerpoint"]),
    "CQP": ("Cheniere Energy Partners", UTILITIES, ["cheniere energy partners", "cheniere partners"]),
    "D": ("Dominion Energy", UTILITIES, ["dominion energy"]),
    "DUK": ("Duke Energy", UTILITIES, ["duke energy"]),
    "ED": ("Consolidated Edison", UTILITIES, ["consolidated edison", "con edison", "coned"]),
    "EIX": ("Edison International", UTILITIES, ["edison international", "southern california edison"]),
    "ES": ("Eversource Energy", UTILITIES, ["eversource"]),
    "ETR": ("Entergy", UTILITIES, ["entergy"]),
    "EXC": ("Exelon", UTILITIES, ["exelon"]),
    "FE": ("FirstEnergy", UTILITIES, ["firstenergy"]),
    "NEE": ("NextEra Energy", UTILITIES, ["nextera energy", "nextera"]),
    "NI": ("NiSource", UTILITIES, ["nisource"]),
    "PCG": ("PG&E", UTILITIES, ["pg&e", "pacific gas and electric"]),
    "PEG": ("Public Service Enterprise Group", UTILITIES, ["public service enterprise group", "pseg"]),
    "PPL": ("PPL", UTILITIES, ["ppl corporation", "ppl corp"]),
    "SO": ("Southern Company", UTILITIES, ["southern company", "southern co"]),
    "VG": ("Venture Global", UTILITIES, ["venture global"]),
    "VST": ("Vistra", UTILITIES, ["vistra"]),
    "WEC": ("WEC Energy Group", UTILITIES, ["wec energy"]),
    "XEL": ("Xcel Energy", UTILITIES, ["xcel energy", "xcel"]),
    # ---------------------------------------------------------------- Healthcare
    "UNH": ("UnitedHealth", HEALTHCARE, ["unitedhealth"]),
    "LLY": ("Eli Lilly", HEALTHCARE, ["eli lilly", "lilly"]),
    "JNJ": ("Johnson & Johnson", HEALTHCARE, ["johnson & johnson", "johnson and johnson", "j&j"]),
    "ABBV": ("AbbVie", HEALTHCARE, ["abbvie"]),
    "MRK": ("Merck", HEALTHCARE, ["merck"]),
    "PFE": ("Pfizer", HEALTHCARE, ["pfizer"]),
    "TMO": ("Thermo Fisher", HEALTHCARE, ["thermo fisher"]),
    "ABT": ("Abbott", HEALTHCARE, ["abbott"]),
    "AMGN": ("Amgen", HEALTHCARE, ["amgen"]),
    "BDX": ("Becton Dickinson", HEALTHCARE, ["becton dickinson", "becton, dickinson"]),
    "BMY": ("Bristol Myers Squibb", HEALTHCARE, ["bristol myers squibb", "bristol-myers squibb", "bristol myers"]),
    "BSX": ("Boston Scientific", HEALTHCARE, ["boston scientific"]),
    "CAH": ("Cardinal Health", HEALTHCARE, ["cardinal health"]),
    "CI": ("Cigna", HEALTHCARE, ["cigna"]),
    "COR": ("Cencora", HEALTHCARE, ["cencora", "amerisourcebergen"]),
    "DHR": ("Danaher", HEALTHCARE, ["danaher"]),
    "GILD": ("Gilead Sciences", HEALTHCARE, ["gilead sciences", "gilead"]),
    "HCA": ("HCA Healthcare", HEALTHCARE, ["hca healthcare"]),
    "HUM": ("Humana", HEALTHCARE, ["humana"]),
    "IDXX": ("IDEXX Laboratories", HEALTHCARE, ["idexx"]),
    "IQV": ("IQVIA", HEALTHCARE, ["iqvia"]),
    "ISRG": ("Intuitive Surgical", HEALTHCARE, ["intuitive surgical"]),
    "MCK": ("McKesson", HEALTHCARE, ["mckesson"]),
    "MDLN": ("Medline", HEALTHCARE, ["medline"]),
    "MDT": ("Medtronic", HEALTHCARE, ["medtronic"]),
    "MRNA": ("Moderna", HEALTHCARE, ["moderna"]),
    "NTRA": ("Natera", HEALTHCARE, ["natera"]),
    "ONC": ("BeOne Medicines", HEALTHCARE, ["beone medicines", "beigene"]),
    "REGN": ("Regeneron", HEALTHCARE, ["regeneron"]),
    "RPRX": ("Royalty Pharma", HEALTHCARE, ["royalty pharma"]),
    "RVMD": ("Revolution Medicines", HEALTHCARE, ["revolution medicines"]),
    "TEVA": ("Teva Pharmaceutical", HEALTHCARE, ["teva pharmaceutical", "teva"]),
    "VRTX": ("Vertex Pharmaceuticals", HEALTHCARE, ["vertex pharmaceuticals", "vertex"]),
    # ---------------------------------------------------------------- Retail
    "AMZN": ("Amazon", RETAIL, ["amazon"]),
    "WMT": ("Walmart", RETAIL, ["walmart", "wal-mart"]),
    "COST": ("Costco", RETAIL, ["costco"]),
    "HD": ("Home Depot", RETAIL, ["home depot"]),
    "MCD": ("McDonald's", RETAIL, ["mcdonald's", "mcdonalds"]),
    "SBUX": ("Starbucks", RETAIL, ["starbucks"]),
    "LOW": ("Lowe's", RETAIL, ["lowe's", "lowes"]),
    "TGT": ("Target", RETAIL, ["target corporation", "target corp"]),
    "AZO": ("AutoZone", RETAIL, ["autozone"]),
    "CMG": ("Chipotle", RETAIL, ["chipotle"]),
    "CPNG": ("Coupang", RETAIL, ["coupang"]),
    "CPRT": ("Copart", RETAIL, ["copart"]),
    "CVNA": ("Carvana", RETAIL, ["carvana"]),
    "CVS": ("CVS Health", RETAIL, ["cvs health", "cvs"]),
    "DG": ("Dollar General", RETAIL, ["dollar general"]),
    "FAST": ("Fastenal", RETAIL, ["fastenal"]),
    "KR": ("Kroger", RETAIL, ["kroger"]),
    "ORLY": ("O'Reilly Automotive", RETAIL, ["o'reilly automotive", "o'reilly auto", "oreilly automotive"]),
    "QSR": ("Restaurant Brands International", RETAIL, ["restaurant brands international", "restaurant brands"]),
    "ROST": ("Ross Stores", RETAIL, ["ross stores"]),
    "SHW": ("Sherwin-Williams", RETAIL, ["sherwin-williams", "sherwin williams"]),
    "TJX": ("TJX Companies", RETAIL, ["tjx"]),
    "WSM": ("Williams-Sonoma", RETAIL, ["williams-sonoma", "williams sonoma"]),
    "YUM": ("Yum! Brands", RETAIL, ["yum! brands", "yum brands"]),
    # ---------------------------------------------------------------- Telecom and media
    "NFLX": ("Netflix", TELECOM, ["netflix"]),
    "DIS": ("Disney", TELECOM, ["disney"]),
    "T": ("AT&T", TELECOM, ["at&t"]),
    "VZ": ("Verizon", TELECOM, ["verizon"]),
    "ASTS": ("AST SpaceMobile", TELECOM, ["ast spacemobile"]),
    "CHTR": ("Charter Communications", TELECOM, ["charter communications"]),
    "ECHO": ("EchoStar", TELECOM, ["echostar"]),
    "FOXA": ("Fox Corporation", TELECOM, ["fox corporation", "fox corp"]),
    "FWONA": ("Liberty Media", TELECOM, ["liberty media"]),
    "GSAT": ("Globalstar", TELECOM, ["globalstar"]),
    "NWSA": ("News Corp", TELECOM, ["news corp"]),
    "ROKU": ("Roku", TELECOM, ["roku"]),
    "SIRI": ("Sirius XM", TELECOM, ["sirius xm", "siriusxm"]),
    "TMUS": ("T-Mobile US", TELECOM, ["t-mobile"]),
    # ---------------------------------------------------------------- Industrials
    "HON": ("Honeywell", INDUSTRIALS, ["honeywell"]),
    "CAT": ("Caterpillar", INDUSTRIALS, ["caterpillar"]),
    "GE": ("GE", INDUSTRIALS, ["general electric", "ge aerospace"]),
    "MMM": ("3M", INDUSTRIALS, ["3m"]),
    "DE": ("Deere", INDUSTRIALS, ["deere"]),
    "LMT": ("Lockheed Martin", INDUSTRIALS, ["lockheed martin", "lockheed"]),
    "RTX": ("RTX", INDUSTRIALS, ["raytheon"]),
    "UPS": ("UPS", INDUSTRIALS, ["united parcel service"]),
    "BA": ("Boeing", INDUSTRIALS, ["boeing"]),
    "CSX": ("CSX", INDUSTRIALS, ["csx corporation", "csx corp"]),
    "ETN": ("Eaton", INDUSTRIALS, ["eaton"]),
    "FDX": ("FedEx", INDUSTRIALS, ["fedex"]),
    "GD": ("General Dynamics", INDUSTRIALS, ["general dynamics"]),
    "ITW": ("Illinois Tool Works", INDUSTRIALS, ["illinois tool works"]),
    "JCI": ("Johnson Controls", INDUSTRIALS, ["johnson controls"]),
    "NOC": ("Northrop Grumman", INDUSTRIALS, ["northrop grumman", "northrop"]),
    "PH": ("Parker-Hannifin", INDUSTRIALS, ["parker-hannifin", "parker hannifin"]),
    "PWR": ("Quanta Services", INDUSTRIALS, ["quanta services"]),
    "RCL": ("Royal Caribbean", INDUSTRIALS, ["royal caribbean"]),
    "RSG": ("Republic Services", INDUSTRIALS, ["republic services"]),
    "TT": ("Trane Technologies", INDUSTRIALS, ["trane technologies", "trane"]),
    "UNP": ("Union Pacific", INDUSTRIALS, ["union pacific"]),
    "WM": ("Waste Management", INDUSTRIALS, ["waste management"]),
    # ---------------------------------------------------------------- Consumer goods
    "PG": ("Procter & Gamble", CONSUMER, ["procter & gamble", "procter and gamble", "p&g"]),
    "KO": ("Coca-Cola", CONSUMER, ["coca-cola", "coca cola", "coke"]),
    "PEP": ("PepsiCo", CONSUMER, ["pepsico", "pepsi"]),
    "TSLA": ("Tesla", CONSUMER, ["tesla"]),
    "NKE": ("Nike", CONSUMER, ["nike"]),
    "BKNG": ("Booking Holdings", CONSUMER, ["booking holdings", "booking.com"]),
    "CL": ("Colgate-Palmolive", CONSUMER, ["colgate-palmolive", "colgate palmolive", "colgate"]),
    "CTAS": ("Cintas", CONSUMER, ["cintas"]),
    "DASH": ("DoorDash", CONSUMER, ["doordash"]),
    "ECL": ("Ecolab", CONSUMER, ["ecolab"]),
    "GM": ("General Motors", CONSUMER, ["general motors"]),
    "HLT": ("Hilton", CONSUMER, ["hilton worldwide", "hilton"]),
    "MAR": ("Marriott", CONSUMER, ["marriott"]),
    "MDLZ": ("Mondelez", CONSUMER, ["mondelez"]),
    "MNST": ("Monster Beverage", CONSUMER, ["monster beverage"]),
    "MO": ("Altria", CONSUMER, ["altria"]),
    "PM": ("Philip Morris International", CONSUMER, ["philip morris"]),
    # ---------------------------------------------------------------- Materials and chemicals
    "APD": ("Air Products", MATERIALS, ["air products"]),
    "CRH": ("CRH", MATERIALS, ["crh plc"]),
    "FCX": ("Freeport-McMoRan", MATERIALS, ["freeport-mcmoran", "freeport mcmoran", "freeport"]),
    "HWM": ("Howmet Aerospace", MATERIALS, ["howmet"]),
    "LIN": ("Linde", MATERIALS, ["linde"]),
    "MLM": ("Martin Marietta Materials", MATERIALS, ["martin marietta"]),
    "NEM": ("Newmont", MATERIALS, ["newmont"]),
    "NUE": ("Nucor", MATERIALS, ["nucor"]),
    "SCCO": ("Southern Copper", MATERIALS, ["southern copper"]),
    "VMC": ("Vulcan Materials", MATERIALS, ["vulcan materials"]),
    # ---------------------------------------------------------------- Real estate
    "AMT": ("American Tower", REAL_ESTATE, ["american tower"]),
    "EQIX": ("Equinix", REAL_ESTATE, ["equinix"]),
    "PLD": ("Prologis", REAL_ESTATE, ["prologis"]),
    "PSA": ("Public Storage", REAL_ESTATE, ["public storage"]),
    "WELL": ("Welltower", REAL_ESTATE, ["welltower"]),
}

# Tickers that are also everyday words or finance abbreviations (AR = accounts receivable, LNG, ICE engines,
# PM, NOW, ALL...). In a question they count only as "$AR" or "ticker AR", like single-letter tickers.
AMBIGUOUS_TICKERS = frozenset(
    "ALL AMP APP AR BA CB CI CL COR DASH DG ECHO ED EG ES FAST FE HOOD ICE KEY LIN LNG MAR MET MO NI NOW "
    "ONC PB PH PM PR RF SO TT WELL WM".split())

NAMES = {ticker: entry[0] for ticker, entry in COMPANIES.items()}
SECTOR_OF = {ticker: entry[1] for ticker, entry in COMPANIES.items()}
ALIASES = {ticker: entry[2] for ticker, entry in COMPANIES.items()}


# Fiscal years named for the year they start in. The corpus labels a filing with the year its fiscal year ends; Home
# Depot and Lowe's call the year that ended in January/February 2026 "fiscal 2025" (their filings print "fiscal 2025"
# in every statement), while Walmart, TJX, NVIDIA and Salesforce call it fiscal 2026. Checked on every filing of the
# index whose year ends in January or February.
NAMED_BY_START_YEAR = frozenset(("HD", "LOW"))


def fiscal_label(ticker: str | None, filing_year: int) -> int:
    """The fiscal year as the company names it, from the corpus's filing year."""
    return filing_year - 1 if str(ticker or "").upper() in NAMED_BY_START_YEAR else filing_year


def filing_year(ticker: str | None, fiscal_year: int | None) -> int | None:
    """The corpus's filing year for a fiscal year as the company names it (None stays None)."""
    if fiscal_year is None:
        return None
    return fiscal_year + 1 if str(ticker or "").upper() in NAMED_BY_START_YEAR else fiscal_year


def normalize_ticker(ticker: str | None) -> str:
    """'brk.b' / 'BRK B' -> 'BRK-B' (the corpus spelling)."""
    return "-".join((ticker or "").strip().upper().replace(".", " ").replace("-", " ").split())


def sector_of(ticker: str | None) -> str | None:
    return SECTOR_OF.get(normalize_ticker(ticker))
