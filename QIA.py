import numpy_financial as npf
import pandas as pd
import numpy as np
import scipy
import math

from scipy import stats
from numpy import log as ln

def create_frequency_table(df, bins, column):
    df = pd.DataFrame(df[column]).groupby(pd.cut(df[column], bins)).count()
    df.columns = ['Frequência Absoluta']
    df['Frequência relativa (%)'] = round(df['Frequência Absoluta'] /
                                    df['Frequência Absoluta'].sum() * 100, 2)
    df['Frequência Absoluta acumulada'] = df['Frequência Absoluta'].cumsum()
    df['Frequência relativa acumulada (%)'] = df['Frequência relativa (%)'].cumsum()
    return df

def compute_continuous_return(pv, fv):
    return ln(fv/pv)
    
def compute_srf(mean, std, minimum_value):
    return (mean - minimum_value) / (std)

def compute_normal_cdf(mu, std, value):
    return scipy.stats.norm(mu, std).cdf(value)

def compute_uniform_cdf_continuous(min_value, max_value, value):
    if value < max_value and value > min_value:
        return (value - min_value)  / (max_value - min_value)
    else:
        return ("Error: value is probably not betwwen the min and max values")
    
def compute_binomial_pmf(p, n, k):
    '''
    p = probabilidade de sucesso
    n = número de tentativas
    k = número de sucessos
    '''
    return scipy.stats.binom.pmf(p=p, n=n, k=k)

def compute_binomial_cdf(p, n, k):
    '''
    p = probabilidade de sucesso
    n = número de tentativas
    k = número de sucessos
    '''
    return scipy.stats.binom.cdf(p=p, n=n, k=k)

def compute_binomial_var(p, n):
    '''
    p = probabilidade de sucesso
    n = número de tentativas
    '''
    return scipy.stats.binom.var(p=p, n=n)

def compute_binomial_std(p, n):
    '''
    p = probabilidade de sucesso
    n = número de tentativas
    '''
    return scipy.stats.binom.std(p=p, n=n)

def compute_FV_ordinary_annuity(A, r, N):

    # A = Pagamento / cash flow
    # r = taxa de juro
    # N = Nr. de anos

    FV = A * (((1+r)**N -1) / r)
    return FV

def compute_FV_lump_sum(PV, rs, N, m=1):

    # FV = Future value / valor futuro
    # PV = Present value / Valor actual
    # rs = Stated rate / Taxa de juro cotada
    # N = número de anos

    FV = PV * (1+rs/m)**(N*m)
    return FV

def compute_PV_lump_sum(FV, rs, N, m=1):

    # PV = Present Value
    # rs = taxa de juro
    # N = Nr de anos/períodos
    # m = 1 para ter capitalizaçãoo anual por defeito

    PV = FV * (1+(rs/m))**(-N*m)
    return PV

def compute_PV_ordinary_annuity(A, r, N):
    
    # A = Pagamento / cash flow
    # r = taxa de juro
    # N = Nr. de anos
    
    PV = A * (1-1/(1+r)**N)/ r
    return PV

def compute_PV_ordinary_annuity_ucf(cf, r, m=1):
    PV_total = 0
    N=0
    for cash_flow in cf:
        N = N + 1
        PV = compute_PV_lump_sum(cash_flow, rs=r, N=N)
        PV_total = PV_total + PV
        
    return PV_total

def compute_pmt(FV, rs, N):
    
    # FV = Future value / valor futuro
    # rs = taxa de juro
    # N = Nr de anos/per�odos
    # pmt = pagamento
    
    pmt = npf.pmt(rate=rs, nper=N, fv=FV, pv=0)
    return pmt

def compute_PV_annuity_due(A, r, N):
    PV = (A * ((1-(1+r)**-N)) / r) * (1+r)
    return PV

def compute_ear(rs, m):

    # FV = Future value / valor futuro
    # PV = Present value / Valor actual
    # rs = Stated rate / Taxa de juro cotada
    # m = n�mero de capitaliza��es anuais

    ear = (1 + rs/m)**m -1
    return ear

def compute_N(FV, PV, r):
    N = ln(FV/PV) / ln(1+r)
    return N

def compute_FV_lump_sum_continuous(PV, rs, N):

    # FV = Future value / valor futuro
    # PV = Present value / Valor actual
    # rs = Stated rate / Taxa de juro cotada
    # N = n�mero de anos

    FV = PV * math.e**(rs*N)
    return FV

def compute_PV_perpetuity(A, r, m=1):
    return A/(r/m)

def compute_FV_ordinary_annuity_ucf(cf, r, N, m=1):
    
    # ucf = unequal cash flows
    # cf = série de cash flows
    # r = taxa de juro
    # N = Nr de anos
    
    FV_total = 0 # criar uma variável para somar os valores futuros dos cash flows
    
    for cash_flow in cf: # Para cada cash flow na série de cash flows
        N = N - 1
        FV = compute_FV_lump_sum(cash_flow, rs=r, N=N) # calcular o cash flow
        FV_total = FV_total + FV # somar cada valor futuro ao valor total
        
    return FV_total

def compute_g(FV, PV, N):
    g = ((FV/PV)**(1/N))-1
    return g

def compute_A(PV, rs, N, m=1):
    PVAF = (1-1/(1+(rs/m))**(m*N))/(rs/m)
    A=PV/PVAF
    return A

def normalize(df):
    df = df.dropna()
    return (df / df.iloc[0]) * 100


def compute_regression_coeficients(x, y):
    return stats.linregress(x=x, y=y)

def compute_regression_value(x, y, x_value):
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    return slope * x_value + intercept

def compute_regression_line(x, y):
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    def compute_regression_value_helper_function(x):
        return slope * x + intercept

    return list(map(compute_regression_value_helper_function, x))

def create_regression_dataframe(x, y):
    regression_line = compute_regression_line(x, y)
    df = pd.DataFrame()

    df['Y'] = list(y)
    df['X'] = list(x)
    df['Regression'] = list(regression_line)

    return df

def compute_coefficient_of_determination(squares_reg, sum_squares_total):
    '''
    squares regression/sum of squares total
    '''
    return squares_reg/sum_squares_total

def compute_std_error_reg(mean_squared_error):
    return np.sqrt(mean_squared_error)

def compute_diference_in_years(start, end):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    difference  = end_date - start_date
    difference_in_years = (difference.days)/365.2421
    return difference_in_years