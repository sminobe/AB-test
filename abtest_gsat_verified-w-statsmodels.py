import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy import stats



def linreg_fit_predict_uncertainty(x,y,x_prdct,alpha=0.05):
    """
    This script calculate linear regression model, built with learning 
    of data of X and y, and predict for X_prdct with uncertainty 
    range (up, lw) of alpha significant (e.g., 1-alpha confident 
    interval of data). 

    This function can be used for the case y is multivariate 
    variable (e.g., time, lat, lon) with a high speed (no loop for 
    lat-lon points). 

    To validate this function, I also made 
        linreg_fit_predict_uncertainty_statsmodels, 
    using a python package statsmodels. These two implementations 
    give identical results for univariate data y
    

    Input data are 
        x: univariate time series eigher (time,) or (time, 1) 
        y: this can be univariate time series (time) or multi-variate 
           data (e.g., (time, lat, lon) or (time, level, lat, lon)
        x_prdct: data points where prediction uncertainty range is estimated with 1-alpha confidence 

    Outputs: 
        a: regression coefficient
        b: intercept
        y_est: estimation of y for X
        y_prdct: prediction of y for X_prdct
        y_prdct_lw: Lower limit of y_prdct 
        y_prdct_up: Upper limit of y_prdct 


    Written by Shoshiro Minobe (Aug 24, 2024)
    """



    y_shape=y.shape
    if len(y_shape)>1:
        y_=y.reshape(y_shape[0],np.prod(y_shape[1:]))
    elif len(y_shape)==1:
        y_=y.reshape(y_shape[0],1)

    x_shape=x.shape
    if len(x_shape)==1:
        x_=x.reshape(x_shape[0],1)
    else:
        x_=x

    xm=x_.mean()
    ym=y_.mean(axis=0)


    xycov=np.sum((x_-xm)*(y_-ym),axis=0)
    xvar =np.sum((x_-xm)**2,axis=0)
    yvar =np.sum((y_-ym)**2,axis=0)

    slope = xycov/xvar
    intcp = ym - slope * xm
    y_est= slope * x_ + intcp


    tlen=y.shape[0]

    se=np.sqrt( np.sum((y_-y_est)**2/ (tlen-2),axis=0))
    tinv=stats.t.ppf(1-alpha/2,tlen)    

    x_prdct_shape=x_prdct.shape
    if len(x_prdct_shape)==0:
        x_prdct_=x_prdct.reshape(1,1)
    elif len(x_prdct_shape)==1:
        x_prdct_=x_prdct.reshape(x_prdct_shape[0],1)
    else:
        x_prdct_=x_prdct


    y_prdct    = slope * x_prdct_ + intcp
    y_prdct_up = y_prdct + tinv *se* np.sqrt(1 + 1/tlen + (x_prdct_-xm)**2/xvar)
    y_prdct_lw = y_prdct - tinv *se* np.sqrt(1 + 1/tlen + (x_prdct_-xm)**2/xvar)


    y_prdct_shape=list(y.shape)
    y_prdct_shape[0] = x_prdct_.shape[0]
    y_prdct    = y_prdct   .reshape(y_prdct_shape)
    y_prdct_lw = y_prdct_lw.reshape(y_prdct_shape)
    y_prdct_up = y_prdct_up.reshape(y_prdct_shape)

    return (slope,intcp,y_est,y_prdct,y_prdct_up,y_prdct_lw)





def linreg_fit_predict_uncertainty_statsmodels(x,y,x_prdct,alpha=0.05):
    """
    This function is a implementation of linreg_fit_predict_uncertainty using a python package, statsmodels. 
    This function and above linreg_fit_predict_uncertainty gives the identical results, confirming the 
    above implementation is OK. This function is used only for univariate y, but the 
    above linreg_fit_predict_uncertainty is used for multivariate y. 

    Input data are 
        x: univariate time series eigher (time,) or (time, 1) 
        y: univariate time series
        x_prdct: data points where prediction uncertainty range is estimated with 1-alpha confidence 

    Outputs: 
        a: regression coefficient
        b: intercept
        y_est: estimation of y for X
        y_prdct: prediction of y for X_prdct
        y_prdct_lw: Lower limit of y_prdct 
        y_prdct_up: Upper limit of y_prdct 


    Written by Shoshiro Minobe (Aug 24, 2024)

    """
    import statsmodels.api as sm
    y_ =y.reshape(y.shape[0])    # statsmodels sm assume y has the shape of (tlen,)
                                 #                       x can have the shape of either (tlen,) or (tlen,1)
#   x_ =x.reshape(x.shape[0],1)

    x_      = sm.add_constant(x)
    model   = sm.OLS(y,x_)
    results = model.fit()
    slope   = results.params[1]
    intcp   = results.params[0]

    pred    = results.get_prediction(x_)
    pred_summary = pred.summary_frame(alpha=alpha)
    y_est   = pred_summary['mean']

    x_prdct_ = sm.add_constant(x_prdct)
    pred     = results.get_prediction(x_prdct_)
    pred_summary = pred.summary_frame(alpha=alpha)
    y_prdct    = pred_summary['mean']
    y_prdct_lw = pred_summary['obs_ci_lower']
    y_prdct_up = pred_summary['obs_ci_upper']

    return (slope,intcp,y_est,y_prdct,y_prdct_up,y_prdct_lw)

def get_yr_mo_dy(xrda_or_xrds):
    """ This function returns integer years, months, days in three numpy arrays.

    input
        xrda: xarray data array or xarray dataset
        time_name: variable name for time. Usually this is time (default),
                  but can be 'T' for example.
    output
        yrs (int):  np array
        mos (int):  np array
        dys (int):  np array
    """

    time_coord=xrda_or_xrds.coords['time']

    yrs=[]
    mos=[]
    dys=[]

    time_index=time_coord.to_index()
    for t in range(len(time_index)):
        time_indx1=time_index[t]
        yr=time_indx1.year
        mo=time_indx1.month
        dy=time_indx1.day
        yrs.append(yr)
        mos.append(mo)
        dys.append(dy)
    yrs=np.array(yrs)
    mos=np.array(mos)
    dys=np.array(dys)
    return yrs,mos,dys

def plot1ssn_yrlytsr_testyrs(xa_yearly_tsr,yr_learn_st=None, alpha=0.05,str_ttl=''):
    """
    plot yearly time series and its points that are significant abnormal record-breaking (i.e., passes AB test)
    along with regression estimation confidence intervals for start of the learning year to the target year. 
    The learning is conducted between the start of the learning year and the one-year before the target year. 

    If the yr_abtest_st is 
    """
    yrs, mos, dys = get_yr_mo_dy(xa_yearly_tsr)
    yrsf =   yrs + (mos-1)/12.+(dys-0.5)/365.
    yr_abtest=yrs[-1]


    plt.plot(yrsf,xa_yearly_tsr,color='black')
    ind_learn_st =np.where(yrs==yr_learn_st)[0][0]

    ind_yr_abtest=np.where(yrs==yr_abtest)[0][0]

    x_train = yrsf[ind_learn_st:ind_yr_abtest]
    y_train = xa_yearly_tsr  [ind_learn_st:ind_yr_abtest].values
    x_prdct = yrsf[ind_learn_st:ind_yr_abtest+1]

    # My function 
    (a1, b1, y_est1, y_prdct1, y_prdct_up1, y_prdct_lw1)=linreg_fit_predict_uncertainty     \
                                                        (x_train,y_train,x_prdct,alpha=alpha)
    # To verify my function using statsmodels
    (a2, b2, y_est2, y_prdct2, y_prdct_up2, y_prdct_lw2 )=linreg_fit_predict_uncertainty_statsmodels \
                                                        (x_train,y_train,x_prdct,alpha=alpha)

    tab10 = plt.get_cmap("tab10") 

    plt.plot(x_prdct, y_prdct_up1, '--', color=tab10(0)) # results of my function are shown by blow dashed lines
    plt.plot(x_prdct, y_prdct_lw1, '--', color=tab10(0))
    plt.plot(x_prdct, y_prdct_up2, '.',  color=tab10(1)) # results of statsmodels are shown by orange dots
    plt.plot(x_prdct, y_prdct_lw2, '.',  color=tab10(1))
    if xa_yearly_tsr[ind_yr_abtest]>y_prdct_up1[-1] or xa_yearly_tsr[ind_yr_abtest]<y_prdct_lw1[-1]:
        plt.plot(yrsf[ind_yr_abtest],xa_yearly_tsr[ind_yr_abtest],'o',label='%i'%yr_abtest,color=tab10(0))
    else:
        plt.plot(yrsf[ind_yr_abtest],xa_yearly_tsr[ind_yr_abtest],'x',label='%i'%yr_abtest,color=tab10(0))

    plt.title(str_ttl)
    return


# main 
xa_t2m_gl_a_1sn=xr.open_dataset('gsat_anom_clm93-22.nc')['t2m']


plt.ion()
plt.figure(1,(12.5,8))
plt.clf()

plot1ssn_yrlytsr_testyrs(xa_t2m_gl_a_1sn, yr_learn_st=1993,alpha=0.10,
                         str_ttl='July-December, Global Ave. SAT Anomaly')
plt.ylabel('($\degree$C)')

