import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from deap import base, creator, tools
import random
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import sys

# Define the portfolio size and weight bounds
portfolio_size = 50
weight_bounds = (0.002, 0.05)

# Function to generate initial weights within constraints
def generate_initial_weights(portfolio_size, min_weight, max_weight, total=1.0):
    """
    Generates a list of weights ensuring each weight is within [min_weight, max_weight] and sums to total.
    """
    weights = []
    remaining_total = total
    for i in range(portfolio_size):
        if i == portfolio_size - 1:
            # Assign the remaining total to the last ticker
            w = remaining_total
        else:
            # Calculate the minimum and maximum possible weight for this ticker
            min_w = max(min_weight, remaining_total - max_weight * (portfolio_size - i - 1))
            max_w = min(max_weight, remaining_total - min_weight * (portfolio_size - i - 1))
            w = random.uniform(min_w, max_w)
        weights.append(w)
        remaining_total -= w
    # Shuffle to randomize distribution
    random.shuffle(weights)
    return weights

# Function to verify and adjust weights
def verify_and_adjust_weights(weights, lower_bound=0.99, upper_bound=1.01):
    """
    Verifies that the sum of weights is within the specified bounds.
    If not, normalizes the weights to sum to 1.

    Args:
        weights (np.ndarray): Array of weights.
        lower_bound (float): Lower bound for the sum of weights.
        upper_bound (float): Upper bound for the sum of weights.

    Returns:
        np.ndarray: Verified and possibly adjusted weights.
    """
    total = weights.sum()
    if lower_bound <= total <= upper_bound:
        return weights
    else:
        print(f"Sum of weights ({total:.4f}) is outside the range [{lower_bound}, {upper_bound}]. Normalizing weights.")
        return weights / total

# Function to fetch monthly prices using Yahoo Finance with a progress bar
def fetch_monthly_prices(tickers, start_date, end_date):
    """
    Fetches monthly adjusted close prices for all tickers using Yahoo Finance.
    Handles missing tickers and unavailable 'Adj Close' data.
    """
    data = {}  # Dictionary to hold the fetched data
    print("Fetching historical data for the given tickers...")

    for ticker in tqdm(tickers, desc="Fetching data from Yahoo Finance", unit="ticker"):
        try:
            # Fetch the data for each ticker
            ticker_data = yf.download(ticker, start=start_date, end=end_date, interval='1mo', progress=False)

            # Check if 'Adj Close' exists, otherwise skip the ticker
            if 'Adj Close' not in ticker_data.columns:
                print(f"'Adj Close' not found for {ticker}, skipping.")
                continue

            # Add the 'Adj Close' data to our dictionary
            data[ticker] = ticker_data['Adj Close']
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Handle the case where no valid data is fetched
    if df.empty:
        print("No valid data fetched from Yahoo Finance.")
        return pd.DataFrame()  # Return an empty DataFrame

    return df


# Genetic Algorithm evaluation function
def eval_sharpe(individual, mean_returns, cov_matrix, risk_free_rate=0.05):
    """
    Evaluate the Sharpe ratio of a portfolio.
    """
    tickers = [t for t, w in individual]  # Extract tickers
    weights = np.array([w for t, w in individual])
    weights /= weights.sum()  # Normalize weights

    # Ensure weights are within bounds
    if not all(weight_bounds[0] <= w <= weight_bounds[1] for w in weights):
        return -np.inf,

    try:
        portfolio_return = np.dot(weights, mean_returns.loc[tickers])
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.loc[tickers, tickers], weights)))
    except KeyError as e:
        # If a ticker is not found in mean_returns or cov_matrix, return invalid fitness
        print(f"KeyError for tickers: {tickers}, Error: {e}")
        return -np.inf,

    if portfolio_volatility == 0:
        return -np.inf,  # Invalid portfolio with zero volatility

    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility  # Risk-free rate
    return sharpe_ratio,

# Custom mutation function to transfer weights between tickers
def mutate_weights(individual, indpb=0.1):
    """
    Mutate the weights of an individual by transferring weight from one ticker to another,
    ensuring weights stay within bounds and sum to 1.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            # Select a ticker to transfer weight from
            source = i
            # Determine how much can be transferred from source
            max_transfer_from = individual[source][1] - weight_bounds[0]
            if max_transfer_from <= 0:
                continue  # Cannot transfer any from this ticker
            # Select a ticker to transfer weight to
            target = random.randint(0, len(individual) - 1)
            while target == source:
                target = random.randint(0, len(individual) - 1)
            # Determine how much can be transferred to target
            max_transfer_to = weight_bounds[1] - individual[target][1]
            if max_transfer_to <= 0:
                continue  # Cannot transfer to this ticker
            # Determine the actual transfer amount
            transfer = random.uniform(0.001, min(max_transfer_from, max_transfer_to))
            # Perform the transfer
            individual[source] = (individual[source][0], individual[source][1] - transfer)
            individual[target] = (individual[target][0], individual[target][1] + transfer)
    return (individual,)

# Function to repair individuals ensuring unique tickers
def repair_individual(individual, available_tickers):
    """
    Ensure that all tickers in the individual are unique.
    Replace duplicates with random available tickers.
    """
    seen = set()
    duplicates = []
    for idx, (ticker, weight) in enumerate(individual):
        if ticker in seen:
            duplicates.append(idx)
        else:
            seen.add(ticker)
    for idx in duplicates:
        # Replace duplicate ticker with a new random ticker not in seen
        new_ticker = random.choice(list(available_tickers - seen))
        individual[idx] = (new_ticker, individual[idx][1])
        seen.add(new_ticker)
    return (individual,)

# Custom crossover function ensuring unique tickers
def custom_crossover(ind1, ind2, available_tickers):
    """
    Custom crossover that ensures tickers remain unique after crossover.
    """
    tools.cxTwoPoint(ind1, ind2)
    # Repair individuals to ensure unique tickers
    repair_individual(ind1, available_tickers)
    repair_individual(ind2, available_tickers)
    return ind1, ind2

# Function to calculate population diversity
def calculate_diversity(population):
    """
    Calculate population diversity based on unique tickers.
    """
    unique_tickers = set()
    for individual in population:
        unique_tickers.update([t for t, w in individual])
    return len(unique_tickers) / (len(population) * portfolio_size)

def main():
    # Define start and end dates for historical data
    start_date = '2023-09-01'
    end_date = '2024-09-28'

    # List of tickers to use (ensure tickers are valid and currently listed)
    tickers = [
        'NVDA', 'CWST', 'FICO', 'PGR', 'CDNS', 'LLY', 'AVGO', 'FIX', 'SNPS', 'MSFT',
        'COST', 'CTAS', 'CPRT', 'MPWR', 'MSCI', 'NOW', 'RSG', 'AJG', 'TMUS', 'UNH',
        'WCN', 'CBZ', 'CELH', 'HEI', 'BRO', 'EME', 'BSX', 'NVMI', 'TT', 'ORLY', 'TTEK',
        'LII', 'TDG', 'BMI', 'MSI', 'MUSA', 'AMZN', 'BAH', 'FTNT', 'DSGX', 'IESC',
        'ONTO', 'CDW', 'INTU', 'ISRG', 'MA', 'AXON', 'TRI', 'AAPL', 'AMD', 'V', 'ANET',
        'AZO', 'BLDR', 'ERIE', 'FN', 'ROL', 'DHR', 'FI', 'INSW', 'NVO', 'SPGI', 'WM',
        'CHDN', 'ENSG', 'KLAC', 'MELI', 'AAON', 'ACGL', 'MMC', 'NDAQ', 'ODFL', 'TSM',
        'ASML', 'CAMT', 'EXLS', 'CACI', 'EXPO', 'GLOB', 'IDXX', 'TYL', 'WSO', 'ADBE',
        'DECK', 'ENTG', 'FCN', 'TMO', 'AFG', 'APH', 'ARES', 'BR', 'CASY', 'GOOG',
        'LRCX', 'PANW', 'POOL', 'ZTS', 'DHI', 'LDOS', 'PWR', 'WRB', 'GOOGL', 'IBP',
        'IT', 'MOH', 'NSSC', 'POST', 'SYK', 'AVY', 'EXEL', 'FSS', 'WST', 'AMAT',
        'GRBK', 'MCO', 'TXRH', 'CSL', 'LOGI', 'TTWO', 'UFPI', 'VRSK', 'ETN', 'MANH',
        'PFSI', 'SPSC', 'VEEV', 'ACLS', 'ELV', 'ICE', 'PHM', 'SHW', 'STE', 'STRL',
        'FLUT', 'SKY', 'SSD', 'TDY', 'TJX', 'ACN', 'CSGP', 'EQIX', 'HD', 'HWKN',
        'MCRI', 'MMSI', 'NFLX', 'PLUS', 'TPL', 'AON', 'ELS', 'LIN', 'LOW', 'NOC',
        'ORCL', 'PAYC', 'PRI', 'SIGI', 'TXN', 'ADP', 'AGYS', 'CBOE', 'CORT', 'DE',
        'FDS', 'GMAB', 'ICLR', 'META', 'RMD', 'SAIA', 'TSCO', 'ADUS', 'AFL', 'AZPN',
        'CME', 'CW', 'DPZ', 'IBKR', 'PH', 'ROP', 'SONY', 'SPXC', 'WD', 'CYBR', 'HCA',
        'KKR', 'RDNT', 'SKYW', 'SMCI', 'VMC', 'WMS', 'ADI', 'AIT', 'CIGI', 'CROX',
        'LVMUY', 'MCD', 'MORN', 'PCTY', 'SCI', 'TER', 'TSLA', 'ANSS', 'BURL', 'CRUS',
        'CYTK', 'JBL', 'LPX', 'LULU', 'NEE', 'OC', 'PTC', 'ASTH', 'GGG', 'GIB', 'ON',
        'RGEN', 'RS', 'SFBS', 'VRNS', 'XPEL', 'BX', 'CENTA', 'CI', 'EA', 'HLT', 'MHO',
        'MLI', 'PAR', 'ROST', 'TREX', 'WMT', 'XYL', 'AEIS', 'CASH', 'CLH', 'CRH',
        'GWW', 'IQV', 'JKHY', 'MGPI', 'NSP', 'NYT', 'PRMW', 'REXR', 'WIX', 'XPO',
        'AME', 'APO', 'BYD', 'CRM', 'EVR', 'GFF', 'LEN', 'MASI', 'MLM', 'NICE', 'PODD',
        'RBC', 'RELX', 'RLI', 'VCEL', 'VRTX', 'ATO', 'AWK', 'CHD', 'COHR', 'FR',
        'MATX', 'MRVL', 'PATK', 'URI', 'ADSK', 'ALL', 'AVAV', 'BLFS', 'IRM', 'JPM',
        'KBH', 'LMT', 'MSA', 'QTWO', 'ROCK', 'A', 'CB', 'CNXN', 'DXCM', 'EGP', 'FORM',
        'LNW', 'LSCC', 'LYV', 'MOD', 'MTH', 'NDSN', 'ORI', 'PAYX', 'PLD', 'PRFT',
        'STLD', 'VIRC', 'VRSN', 'AIZ', 'CMG', 'CPK', 'DY', 'EFX', 'EXR', 'FERG',
        'GRMN', 'GWRE', 'HALO', 'HCKT', 'ITW', 'OLED', 'QLYS', 'ZBRA', 'ABBV', 'ADC',
        'AOS', 'BCO', 'BRC', 'BWXT', 'CBRE', 'CCS', 'CNC', 'COR', 'EW', 'LGIH', 'LPLA',
        'MKTX', 'MNST', 'PCAR', 'PKG', 'PLXS', 'TOELY', 'TOL', 'ABT', 'ATR', 'CUBE',
        'EXPD', 'FAST', 'GPI', 'HIG', 'HOLX', 'MAS', 'MSTR', 'NTES', 'OTTR', 'SUI',
        'TBBK', 'AMP', 'ASB', 'AWR', 'CAT', 'CHKP', 'CZR', 'DKS', 'FBP', 'IEX', 'LRN',
        'MSEX', 'MYRG', 'NMIH', 'SSNC', 'TGLS', 'TGS', 'WAT', 'ACMR', 'AGO', 'ALLE',
        'AMPH', 'BANF', 'BKNG', 'BPOP', 'CSX', 'GMED', 'HTHT', 'J', 'LSTR', 'SAIC',
        'SLP', 'STZ', 'TSEM', 'TTC', 'WSM', 'ALSN', 'BCC', 'BRKR', 'CINF', 'DRI',
        'FLEX', 'INFY', 'NXST', 'OSBC', 'PAG', 'RUSHA', 'SAP', 'SFM', 'WTW', 'AXP',
        'AZTA', 'BN', 'COO', 'CRL', 'DOV', 'EPAM', 'IBN', 'LRLCY', 'MGRC', 'SCCO',
        'SCVL', 'TMHC', 'TRV', 'WWD', 'ALGN', 'ASGN', 'BYDDY', 'CLS', 'EFSC',
        'FELE', 'FNF', 'GPN', 'HQY', 'IAC', 'IDCC', 'KBR', 'MAR', 'MTRN', 'MURGY',
        'PRDO', 'REGN', 'RNR', 'SKX', 'SNX', 'TECH', 'THG', 'AMH', 'BFAM', 'CMS',
        'FIVN', 'MAA', 'MCHP', 'MKC', 'MTSI', 'NXPI', 'PEGA', 'STM', 'TCEHY',
        'TNET', 'TRUP', 'YUM', 'AMRC', 'AMT', 'BLK', 'CIEN', 'CPRX', 'CVLT', 'DLR',
        'DORM', 'ESNT', 'ITT', 'JBT', 'LAD', 'LECO', 'MCK', 'MPC', 'MS', 'ORA',
        'POWL', 'QNST', 'RPM', 'SANM', 'STEL', 'TPH', 'ULTA', 'ABG', 'ACM', 'AN',
        'BIO', 'CTS', 'CWT', 'DGII', 'GD', 'HROW', 'IHG', 'INSM', 'KFRC', 'LHX',
        'MRCY', 'MTG', 'MTZ', 'RGA', 'RTX', 'SBUX', 'SUPN', 'TEL', 'TPX', 'AEE',
        'AGCO', 'AX', 'BECN', 'DGX', 'ENPH', 'FSUGY', 'GDEN', 'HDB', 'JOE', 'LKFN',
        'MKSI', 'MMYT', 'PEG', 'RICK', 'RMBS', 'ABBNY', 'AGX', 'BBY', 'BERY', 'CMC',
        'EG', 'FFIN', 'GNRC', 'GPK', 'GS', 'HII', 'IFNNY', 'IRDM', 'ITGR', 'LAMR',
        'PSA', 'SF', 'WDAY', 'WEC', 'WPM', 'ABCB', 'ACIW', 'AMGN', 'AWI', 'BYRN',
        'CALX', 'EBAY', 'EHC', 'ERII', 'EXP', 'FAF', 'FRO', 'FSLR', 'ICUI', 'ITCI',
        'KFY', 'KTOS', 'LNT', 'MDLZ', 'NTAP', 'PG', 'SRPT', 'UCTT', 'WAL', 'AMKR',
        'AMN', 'ANF', 'AOSL', 'FBNC', 'FNV', 'G', 'GATX', 'IBOC', 'KLIC', 'LCII',
        'MRTN', 'NI', 'OFG', 'PRGS', 'RH', 'SEM', 'STC', 'TOWN', 'UNP', 'UTHR', 'XEL',
        # Added tickers
        'AIN', 'ALNY', 'APD', 'AZN', 'CHH', 'ECL', 'HCI', 'HON', 'HSY', 'HUM', 'MWA',
        'PNFP', 'SJW', 'SLAB', 'SLGN', 'SNA', 'TTGT', 'VLO', 'WNS', 'AEM', 'AIR',
        'ARW', 'CMI', 'CNS', 'COLM', 'COOP', 'CRS', 'CTO', 'JBHT', 'JEF', 'KNTK',
        'MMS', 'NEO', 'NRG', 'NUE', 'OGS', 'PBH', 'PLAB', 'ROK', 'SBCF', 'TFX', 'TGT',
        'TXT', 'VCYT', 'WH', 'ARCB', 'CDNA', 'DIOD', 'EXAS', 'GLW', 'H', 'HTLF',
        'ITRI', 'LH', 'RDN', 'SBAC', 'SLM', 'SO', 'TTMI', 'WTFC', 'AEP', 'AORT', 'BAC',
        'BDX', 'CP', 'FBMS', 'GNTX', 'HWM', 'KWR', 'LEU', 'MU', 'NBHC', 'NSC', 'SONVY',
        'VCISY', 'BALL', 'CHTR', 'CMPGY', 'CTRE', 'CUBI', 'CWCO', 'DTE', 'DVA', 'EWBC',
        'FIVE', 'FOXF', 'GVA', 'HASI', 'HUBG', 'IRT', 'KR', 'LNG', 'MRK', 'NEM', 'OSIS',
        'OSK', 'PEP', 'PNC', 'RJF', 'STAA', 'THC', 'UHS', 'USPH', 'VRDN', 'WEN', 'WGO',
        'WLDN', 'AGI', 'AMWD', 'AYI', 'BLKB', 'CCJ', 'CCOI', 'CE', 'CPAY', 'CSCO', 'CTLT',
        'DTEGY', 'EADSY', 'EPC', 'GGAL', 'GL', 'IDA', 'LYTS', 'POWI', 'RF', 'SCHW',
        'SHOO', 'UMH', 'WSFS', 'AER', 'CCK', 'CDPYF', 'CIHKY', 'CPT', 'DOX', 'FCFS',
        'GTY', 'HAE', 'HPQ', 'IDEXY', 'LZB', 'MTN', 'NBTB', 'PDFS', 'QCOM', 'QGEN',
        'RCL', 'RNG', 'SYY', 'TECK', 'VMI', 'WAB', 'WELL', 'ABM', 'ATSG', 'AU', 'BIP',
        'CBT', 'CGNX', 'CHEF', 'CNO', 'DFS', 'DOOO', 'EDU', 'EQC', 'FANG', 'FCF', 'FITB',
        'FUL', 'GEN', 'HRB', 'L', 'MC', 'NJR', 'RGLD', 'RHP', 'STAG', 'XNCR', 'ARMK',
        'CNMD', 'ED', 'EEFT', 'JLL', 'PRIM', 'SA', 'SBS', 'SIMO', 'SRDX', 'VOYA', 'WCC',
        'BHE', 'CCEP', 'CECO', 'COHU', 'GLPG', 'HZO', 'KNX', 'MBUU', 'SEIC', 'TRMB',
        'UMBF', 'WOR', 'ANIP', 'AVNW', 'AZZ', 'BC', 'BK', 'BZH', 'CSGS', 'CSV', 'DAR',
        'EDN', 'FRME', 'HOMB', 'INGR', 'SNV', 'STBA', 'SYF', 'TGTX', 'THO', 'TKR', 'ATRC',
        'CVEO', 'DCI', 'ESS', 'FDX', 'FFIV', 'GBCI', 'GTLS', 'HSII', 'MAIN', 'MDGL',
        'NKE', 'PAAS', 'PFC', 'ROG', 'RYAAY', 'TNK', 'WAFD', 'CATY', 'CBU', 'CEVA',
        'CLX', 'CNI', 'CTSH', 'DCOM', 'DLB', 'FOR', 'JNPR', 'KMPR', 'KO', 'MTDR', 'PPBI',
        'RY', 'SSB', 'STX', 'UDR', 'UTI', 'VNOM', 'WERN', 'WEX', 'ZEUS', 'ADMA', 'APOG',
        'ARE', 'ASH', 'ATGE', 'AXS', 'CAR', 'CFG', 'COF', 'DQ', 'ENS', 'FULT', 'GLPI',
        'GME', 'INDB', 'IPG', 'KB', 'KMX', 'NWS', 'OLN', 'PENN', 'PNW', 'QDEL',
        'SFTBY', 'UBS', 'WTRG', 'ACHC', 'AKAM', 'ASC', 'ATI', 'AUB', 'AVB', 'BMA', 'CNNE',
        'CNQ', 'EMR', 'ETR', 'EXC', 'GIL', 'JCI', 'OMCL', 'SWKS', 'TCOM', 'TEX', 'TREE',
        'TSLX', 'UNM', 'USM', 'UVE', 'WLK', 'ZG', 'AGR', 'AMSC', 'BDC', 'BEP', 'BOKF',
        'BUSE', 'CL', 'FFBC', 'HBNC', 'HTH', 'MET', 'MGM', 'NWSA', 'RAMP', 'SQM', 'STRA',
        'ZION', 'ALB', 'ARWR', 'CALM', 'CMPR', 'CPF', 'EXPE', 'GPC', 'HXL', 'KAR', 'MIDD',
        'NX', 'OMF', 'PFG', 'PSX', 'R', 'RYI', 'SMCAY', 'SMFNF', 'SMTC', 'SRE', 'TSN',
        'WBS', 'BANR', 'CAE', 'CAH', 'COP', 'DUK', 'EAT', 'EIX', 'ES', 'EVTC', 'FIS',
        'HSIC', 'JNJ', 'LKQ', 'NVT', 'O', 'OKE', 'PBF', 'RDY', 'SIEGY', 'SLF', 'TX', 'UL',
        'WNC', 'ALLY', 'CF', 'CHRW', 'CQP', 'ENV', 'EOG', 'FCX', 'GE', 'GM', 'GOLD', 'HA',
        'HNI', 'IMO', 'NHI', 'PM', 'POR', 'SAVA', 'SWX', 'UVSP', 'ADX', 'AEO', 'AL',
        'AMBA', 'CCI', 'CFR', 'CG', 'CMCO', 'CMCSA', 'HES', 'IGT', 'MFC', 'OMC', 'ONB',
        'OXM', 'PCH', 'PLOW', 'PNR', 'TM', 'AIG', 'AVNT', 'CAL', 'CRTO', 'CSQ', 'DAC',
        'DG', 'EPAC', 'ETD', 'GIS', 'HFWA', 'HTGC', 'INCY', 'ING', 'KELYA', 'MDT', 'MTB',
        'NNN', 'OPCH', 'OZK', 'PPC', 'SASR', 'SCSC', 'SHECY', 'SMG', 'TRGP', 'TRMK',
        'TROW', 'AAGIY', 'AES', 'ALE', 'CHUY', 'CM', 'CNOB', 'DAL', 'FHN', 'HEES',
        'HKXCY', 'IBTX', 'PARR', 'PRTA', 'PRU', 'PZZA', 'REG', 'SAM', 'SR', 'TCBI',
        'TILE', 'TPR', 'AB', 'B', 'BA', 'BRX', 'CBSH', 'CGEMY', 'CODI', 'DLTR', 'EMN',
        'EOI', 'GLAD', 'HVT', 'INVA', 'KOP', 'RARE', 'RHI', 'RNP', 'SON', 'VSH', 'ARCC',
        'BKU', 'CABGY', 'CMA', 'DAY', 'DKL', 'FDUS', 'FMC', 'GBOOY', 'HAS', 'KEY',
        'NTRS', 'PB', 'PPRUY', 'RIO', 'UPS', 'VAC', 'WMMVY', 'X', 'AVT', 'BHLB',
        'CVBF', 'HLIO', 'IONS', 'JD', 'LEA', 'MHK', 'PPG', 'PTCT', 'RNST', 'SPTN', 'SYNA',
        'TDS', 'THRM', 'UAL', 'VTR', 'ZWS', 'CAG', 'CNA', 'DINO', 'FE', 'FHI', 'HELE',
        'NSRGY', 'RDWR', 'RL', 'SIG', 'STR', 'STT', 'TPC', 'UBSI', 'WY', 'AVA', 'BAP',
        'BLMN', 'BNPQY', 'BOH', 'C', 'CLMT', 'CNK', 'CTRN', 'CVX', 'ESI', 'FIBK', 'FLO',
        'FOXA', 'GIII', 'GSL', 'HMST', 'K', 'KMB', 'KRG', 'MCY', 'OEC', 'OHI', 'SKT',
        'SM', 'SPB', 'SPOK', 'SWK', 'URBN', 'USAC', 'UVV', 'XOM', 'ADM', 'ALV',
        'BG', 'BJRI', 'BKE', 'BKH', 'CNX', 'DD', 'EL', 'EMLAF', 'EPR', 'EQR', 'ERJ',
        'GES', 'HRL', 'IPGP', 'KIM', 'KMTUY', 'OCFC', 'OSPN', 'PFS', 'PRNDY', 'ROIC',
        'SPG', 'SPR', 'TD', 'TFC', 'VC', 'AA', 'ALK', 'BABA', 'BMO', 'CII', 'CSTM',
        'FOX', 'FUN', 'HEINY', 'IBM', 'JKS', 'MANU', 'MLKN', 'MRO', 'OTEX', 'PAHC',
        'SFNC', 'STNG', 'SU', 'CPA', 'EVT', 'FMX', 'IFF', 'KEX', 'MAN', 'NVS', 'NWE',
        'RGR', 'SJM', 'SUN', 'VECO', 'YPF', 'ZUMZ', 'ACAD', 'AGIO', 'ASTE', 'BMRN',
        'CNP', 'COLB', 'CPB', 'CVE', 'EQNR', 'HAFC', 'HMN', 'IP', 'LAZ', 'MMM', 'MUR',
        'NAVI', 'PSMT', 'PVH', 'TR', 'USB', 'WFC', 'XNGSY', 'AR', 'BHP', 'CVI', 'DIS',
        'FLR', 'HUN', 'KALU', 'MO', 'MYGN', 'PBA', 'PHG', 'STWD', 'TEN', 'TRS', 'TTE',
        'VRNT', 'WES', 'ZBH', 'ZD', 'BHPLF', 'CAKE', 'DVN', 'EQT', 'ETG', 'FJTSY',
        'NCLH', 'ODP', 'SEE', 'TRUMY', 'WSBC', 'APEI', 'ATLKY', 'BAX', 'BRDCY', 'CSWC',
        'DASTY', 'FDP', 'FWRD', 'GDV', 'ILMN', 'IMAX', 'LUV', 'PFE', 'PRO', 'RVT',
        'SLG', 'TNDM', 'VRE', 'AKR', 'ALKS', 'AMG', 'ATHM', 'CPRI', 'DEO', 'EGO', 'FRT',
        'GBX', 'HI', 'LE', 'LTC', 'MARA', 'MGA', 'MSM', 'OMI', 'OVV', 'RYN', 'SMP',
        'SSEZY', 'TRMD', 'UTF', 'APAM', 'BMY', 'BSAC', 'CLDX', 'CRI', 'HTD', 'MEOH',
        'NVGS', 'OGE', 'OXY', 'RCI', 'WMB', 'ANDE', 'APA', 'ARCT', 'AROC', 'DK',
        'GAP', 'GOOD', 'KT', 'LYB', 'MDU', 'MED', 'NEP', 'OII', 'PPL', 'RRC', 'TRN',
        'WDC', 'AAT', 'ARLP', 'ET', 'GLNG', 'JACK', 'MPLX', 'NTGR', 'SBLK', 'SBRA',
        'SKM', 'VTOL', 'WPC', 'CCL', 'CVGW', 'HOG', 'HST', 'MOV', 'MT', 'SNY', 'TDC',
        'UTG', 'BBWI', 'CLW', 'CTRA', 'CUK', 'EGBN', 'FL', 'GDS', 'KMT', 'KOF', 'LNC',
        'MAT', 'MOS', 'NEOG', 'PCG', 'SHEL', 'SNN', 'ST', 'TS', 'WHR', 'ATRO', 'FTI',
        'GCO', 'HAL', 'HIW', 'NOG', 'RHHBY', 'SSTK', 'TRP', 'YELP', 'YY', 'FLS',
        'GSK', 'HMC', 'HSBC', 'LVS', 'MTUS', 'NWN', 'TIMB', 'BKR', 'BNS', 'BXMT',
        'CIB', 'ENB', 'GBDC', 'IART', 'KN', 'PII', 'PRAA', 'TLK', 'BWA', 'DIN',
        'ENGIY', 'JWN', 'SLRC', 'TU', 'WYNN', 'BXP', 'NMM', 'PRLB', 'SIRI', 'TAP',
        'UNFI', 'ALEX', 'AY', 'E', 'JAZZ', 'KSS', 'NFG', 'NGG', 'OFIX', 'SATS',
        'BIIB', 'BP', 'EPD', 'GILD', 'INTC', 'MAC', 'TOYOF', 'AMX', 'CVS', 'DB',
        'FMS', 'HP', 'T', 'THS', 'HR', 'IMBBY', 'SLB', 'TEVA', 'UGI', 'VZ', 'ALGT',
        'BIDU', 'BTI', 'DEI', 'FARO', 'MNRO', 'PDCO', 'CUZ', 'D', 'KRC', 'RIOCF',
        'BUD', 'FFC', 'IVZ', 'OCSL', 'SRCL', 'CIM', 'KMI', 'NBR', 'PAA', 'SCHYY',
        'VTLE', 'XRAY', 'BCH', 'FPF', 'PINC', 'TKPHF', 'VNO', 'CBRL', 'LBTYK',
        'NLY', 'WDS', 'DOC', 'HQH', 'LBTYA', 'NXP', 'AAP', 'IFN', 'CLB', 'DLX',
        'PAGP', 'LDP', 'NTCT', 'PUK', 'BBN', 'BCE', 'GNK', 'NOV', 'SVNDY', 'TDW',
        'XPRO', 'BEN', 'NBB', 'SPH', 'VFC', 'GOF', 'PDI', 'PRGO', 'ARR', 'FAX'
    ]

    # Fetch data from Yahoo Finance
    data = fetch_monthly_prices(tickers, start_date, end_date)

    # Check if data was fetched
    if data.empty:
        print("No valid data fetched. Exiting.")
        sys.exit()

    # Calculate monthly returns
    monthly_returns = data.pct_change().dropna()

    # Handle NaNs (drop tickers with excessive NaNs)
    max_nan_fraction = 0.1  # Allow tickers with up to 10% missing data
    acceptable_tickers = monthly_returns.columns[monthly_returns.isna().mean() < max_nan_fraction]
    monthly_returns = monthly_returns[acceptable_tickers]

    # Calculate annualized mean returns and covariance matrix
    mean_returns = monthly_returns.mean(skipna=True) * 12  # Annualized returns
    cov_matrix = monthly_returns.cov(min_periods=12) * 12  # Annualized covariance matrix

    # Remove tickers with NaN in mean returns
    mean_returns = mean_returns.dropna()
    cov_matrix = cov_matrix.loc[mean_returns.index, mean_returns.index]

    # Ensure we have enough tickers
    if len(mean_returns) < portfolio_size:
        print(f"Not enough valid tickers. Required: {portfolio_size}, Available: {len(mean_returns)}.")
        sys.exit()

    # Initialize Genetic Algorithm (DEAP)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Register individual and population creation functions
    def create_individual():
        tickers_selected = random.sample(list(mean_returns.index), portfolio_size)
        weights = generate_initial_weights(portfolio_size, weight_bounds[0], weight_bounds[1], total=1.0)
        return creator.Individual(list(zip(tickers_selected, weights)))

    # Utilize multiprocessing for parallel fitness evaluations
    pool = Pool()
    toolbox.register("map", pool.map)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register Genetic Algorithm operators
    available_tickers = set(mean_returns.index)
    toolbox.register("evaluate", eval_sharpe, mean_returns=mean_returns, cov_matrix=cov_matrix)
    toolbox.register("mate", partial(custom_crossover, available_tickers=available_tickers))
    toolbox.register("mutate", mutate_weights, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Genetic Algorithm Parameters
    population_size = 300  # Increased from 100 to 300
    num_generations = 300   # Increased from 100 to 300
    crossover_probability = 0.8  # Increased from 0.7 to 0.8
    mutation_probability = 0.3   # Increased from 0.2 to 0.3
    elitism_size = 5             # Number of top individuals to retain each generation

    # Create initial population
    population = toolbox.population(n=population_size)

    # Run the Genetic Algorithm with a progress bar
    best_sharpe = -np.inf
    best_individual = None

    best_return = -np.inf  # Initialize best expected return
    best_return_individual = None  # Initialize individual with best expected return

    print("Running Genetic Algorithm...")
    for gen in tqdm(range(num_generations), desc="Running Genetic Algorithm", unit="gen"):
        # Evaluate fitness
        fitnesses = list(toolbox.map(toolbox.evaluate, population))

        # Assign fitness values and keep track of the best individuals
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            if fit[0] > best_sharpe:
                # Verify weight constraints before considering as best
                weights = np.array([w for t, w in ind])
                if all(weight_bounds[0] <= w <= weight_bounds[1] for w in weights):
                    best_sharpe = fit[0]
                    best_individual = ind

            # Compute portfolio return for tracking the best expected return
            weights = np.array([w for t, w in ind])
            portfolio_return = np.dot(weights, mean_returns.loc[[t for t, w in ind]])

            if portfolio_return > best_return:
                if all(weight_bounds[0] <= w <= weight_bounds[1] for w in weights):
                    best_return = portfolio_return
                    best_return_individual = ind

        # Select the next generation individuals excluding elitism
        offspring = toolbox.select(population, len(population) - elitism_size)
        offspring = list(map(toolbox.clone, offspring))

        # Extract the best individuals for elitism
        top_individuals = tools.selBest(population, elitism_size)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_probability:
                toolbox.mate(child1, child2)
                # After mating, fitness is invalidated
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Add elitism back to the offspring
        offspring += top_individuals

        # Replace population with the offspring
        population[:] = offspring

        # Optional: Adjust mutation probability based on diversity
        diversity = calculate_diversity(population)
        if diversity < 0.5:
            mutation_probability = 0.4  # Increase mutation rate
        else:
            mutation_probability = 0.3  # Reset to default

    # Close the multiprocessing pool
    pool.close()
    pool.join()

    # Output the best results
    if best_individual:
        # Extract weights
        weights = np.array([w for t, w in best_individual])

        # Verify weight constraints
        assert all(weight_bounds[0] <= w <= weight_bounds[1] for w in weights), "Weight constraints violated in the best individual."

        # Verify and adjust weights sum
        weights = verify_and_adjust_weights(weights)

        # Optionally, update the individual with adjusted weights
        if not np.isclose(weights.sum(), 1.0):
            best_individual = list(zip([t for t, _ in best_individual], weights))

        print(f"\nBest Sharpe Ratio: {best_sharpe:.4f}")
        print("\nPortfolio with Best Sharpe Ratio = {")
        for ticker, weight in best_individual:
            print(f"    '{ticker}': {weight:.4f},")
        print("}")
        print(f"Sum of weights: {weights.sum():.4f}")
    else:
        print("No optimal portfolio found based on Sharpe Ratio.")

    if best_return_individual:
        # Extract weights
        weights = np.array([w for t, w in best_return_individual])

        # Verify weight constraints
        assert all(weight_bounds[0] <= w <= weight_bounds[1] for w in weights), "Weight constraints violated in the best return individual."

        # Verify and adjust weights sum
        weights = verify_and_adjust_weights(weights)

        # Optionally, update the individual with adjusted weights
        if not np.isclose(weights.sum(), 1.0):
            best_return_individual = list(zip([t for t, _ in best_return_individual], weights))

        print(f"\nBest Expected Return: {best_return:.4f}")
        print("\nPortfolio = {")
        for ticker, weight in best_return_individual:
            print(f"    '{ticker}': {weight:.4f},")
        print("}")
        print(f"Sum of weights: {weights.sum():.4f}")
    else:
        print("No optimal portfolio found based on Expected Return.")

if __name__ == "__main__":
    main()
