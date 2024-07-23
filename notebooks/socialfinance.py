# socialfinance.py  -- module for modeling contracts and bank funding structures

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display, Math


class Bank(object):
    ''' A Bank in a 'neighborhood' or 'zone'  where the representative 
        borrower has pledgeable assets A.  The bank will have (derived) 
        attributes including the terms of contract, monitoring intensity, 
        and its own funding structure.
    '''

    def __init__(self, A, beta): 
        self.A = A         # pledgeable assets (as array)
        self.gamma = 1.0   # cost of uninformed capital (1 + ru)
        self.beta = beta   # cost of equity capital (1 + re)
        self.B0 = 30       # intercept monitoring intensity function
        self.alpha = 0.5   # slope monitoring intensity function
        self.X = 200       # project success return
        self.I = 100       # lump-sum investment
        self.p = 0.97      # prob. of success if diligent
        self.q = 0.82      # prob. of success if not-diligent
        self.F = 0         # Fixed cost per neighborhood
        self.f = 30        # Fixed cost per loan
        self.K = 12000     # Intermediary capital in each neighborhood.
        #self.M = self.minmon(A)
        self.Amax = 140    # used for plot limits  

    def B(self, m):
        '''Monitoring intensity function: B(m) is the private benefit borrower 
        could capture via non-diligence'''
        return self.B0 - self.alpha * m
    
    def FC(self, N):  # Avg fixed cost per borrower if bank has N borrowers
        return self.F / N + self.f

    def AMe(self, m): 
        '''Minimum collateral for non-leveraged or equity-only MFI '''
        p, q, I, X, beta,f = self.p, self.q, self.I, self.X, self.beta, self.f    
        return (p/(p-q)) * self.B(m)  - (p * X - beta * I) + m + f

    def AM(self, m):
        '''Minimum collateral for leveraged MFI '''
        p, q, I, X, gam, beta, f= self.p, self.q, self.I, self.X, self.gamma, self.beta, self.f  
   
        return (p/(p-q)) * self.B(m) - (p * X - gam * I) + m  \
                  + ((beta - gam) / beta) * (q * m / (p - q)) + f

    def Abest(self, m):
        '''Lower of the two collateral requirements'''
        return np.minimum(self.AMe(m), self.AM(m))

    def Im(self, m):
        '''Minimum required equity investment by monitor'''
        return (1/self.beta) * self.q * m / (self.p - self.q)

    def mcross(self):
        '''Monitoring level where equity only AMe and levered AM lines cross'''
        return self.beta * self.I * (self.p - self.q) / self.q

    def Across(self):
        return self.AM(self.mcross())

    def mmax(self):
        '''Maximal monitoring at which equity-only monitor can just break even'''
        return self.p * self.X - self.beta * self.I - self.f

    def Amin(self):
        '''Lowest possible collateral requirement - at max feasible monitoring'''
        return self.AMe(self.mmax())

    def mon(self, A):
        '''optimal monitoring in leveraged MFI
           Zero if >A(0)'''
        AHI = self.AM(0) 
        return ( (AHI - A) * (self.beta * (self.p - self.q)) / 
                 ((self.alpha - 1) * self.beta * self.p + self.gamma * self.q)   )

    def monE(self, A):
        '''optimal monitoring in equity-only MFI'''
        AHI = self.AMe(0)
        return ( (AHI - A) * 
                ((self.p - self.q) / (self.q + (self.alpha-1) * self.p))   )

    def minmon(self, A):
        return np.minimum(self.monE(A), self.mon(A))

    def breturn(self, A):
        ''' array of borrower returns by A'''
        X, p, q, I, f, gam, beta = self.X, self.p, self.q, self.I, self.f, self.gamma, self.beta

        br = [p * X - gam * I - f if a > self.AM(0) 
            else p * X - gam * I - f - self.mon(a) * (1 + ((beta - gam) / beta) * (q / (p - q))) if (a <= self.AM(0)) and (a > self.Across()) 
            else p * X - beta * I - f - self.monE(a) if (a <= self.Across()) and (a >= self.Amin()) 
            else 0 
            for a in A]
        return np.array(br)


    def print_params(self):
        """
        Display scalar parameters alphabetically
        """
        params = sorted(vars(self).items())
        params_to_print = [f"{key} = {value}" for key, value in params if np.isscalar(value)]
        print(', '.join(params_to_print))

    

    def nreach(self,A):
        '''number of borrowers reached with K of intermediary capital at different A'''
        K, F, I = self.K, self.F, self.I

        nr = np.zeros(len(A))
        for i, a in enumerate(A):
            if a > self.AM(0):
                nr[i] = np.nan
            elif (a <= self.AM(0)) and (a > self.Across()):
                nr[i] = K/( self.Im(self.mon(a)) -F )
            elif (a <= self.Across()) and (a >= self.Amin()):
                nr[i] = K/(I+F)
            else:
                nr[i] = 0
        return nr
    
    def plotA(self):
        '''Plot minimum collateral requirements'''
        mc, mx = self.mcross(), self.mmax()
        Am0, Amc, Amx = self.AM(0), self.AM(mc), self.AMe(mx)
        mm, mm_ = np.linspace(0, self.Amax), np.linspace(0, mx)
        
        fig, ax = plt.subplots(1)
        ax.plot(mm, self.AMe(mm), label='equity only MFI', linestyle=':')
        ax.plot(mm, self.AM(mm), label='leveraged MFI', linestyle=':')
        ax.plot(mm_, self.Abest(mm_), linewidth=3.3) 
        
        ax.set(xlim=(0, 80), ylim=(0, self.AMe(0)), 
               title='Minimum Collateral requirement', 
               xlabel='monitoring intensity $m$', ylabel='pledgeable asset $A (m)$')
        
        ax.text(1, Am0+5, 'No monitor', rotation='vertical', verticalalignment='bottom')
        ax.text(1, (Amc+Am0)/2, 'Interme-\n diated', rotation='vertical', verticalalignment='center')
        ax.text(1, (Amx+Amc)/2, 'Equity-only', rotation='vertical', verticalalignment='center')
        ax.text(1, Amx/2, 'No Loan', rotation='vertical', verticalalignment='center')
        ax.text(mx*1.1, self.AMe(mx)*0.9, r'$AM_e(m)$')
        ax.text(mx*1.1, self.AM(mx), r'$AM(m)$')
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.vlines([mc, mx], ymin=0, ymax=[Amc, Amx], linestyle =':')
        ax.hlines([Amc, Amx], xmin=0, xmax=[mc, mx], linestyle =':')
        
        ax.legend(loc='upper right')
        ax.set_ylim(0, self.AMe(0)+20)
    
    def plotIm(self):
        '''
        plot total investment share by intermediary and uninformed lenders'''
        I, f, beta = self.I, self.f, self.beta
        mc, mx = self.mcross(), self.mmax()
        Amc, Amx = self.AM(mc), self.Amax

        amin = self.Amin()
        A_ = np.linspace(amin, Amx, 100)  # color only loans
        Im = np.minimum(I, self.minmon(A_) * (self.q / (self.p - self.q)) / beta )
        Im1 = np.minimum(I, self.mon(A_) * (self.q / (self.p - self.q)) / beta )
        #Im2 = np.minimum(I+f, self.monE(A_) * (self.q / (self.p - self.q)) / beta)


        fig, ax = plt.subplots()
        ax.plot(A_, Im, label=r'$I^m$ - monitoring equity')
        ax.plot(A_, np.minimum(I - Im, I), label=r'$I^u$ - uninformed debt')
        ax.plot(A_, self.minmon(A_), label=r'$m$ - monitoring');

  
        ax.plot(A_, Im1, label=r'$m$ - monitoring')
        #ax.plot(A_, Im2, label=r'$m$ - monitoring')

        ax.set_title(r'Required monitoring m and investment $I_m$')
        ax.set_xlabel('A -- pledgeable assets')
        ax.set_ylim(0, I  + 10)
        ax.set_xlim(amin-10, max(A_))

        ax.text(amin - 5, I, r'$I$')
        ax.text(amin + 2, I, r'$I^m$')
        ax.text(amin + 2, self.monE(amin), 'm(A)')
        ax.text(amin + 2, 2, r'$I^u =I-I^m$')

        ax.axvline(x=amin, linestyle=':')
        ax.axvline(x=Amc, linestyle=':')
        ax.axvline(x=self.AM(0), linestyle=':')
        ax.axhline(y=I , linestyle=':')


    def plotDE(self,beta):
        '''plot outside debt to monitored debt (I+F-Im)/Im ratio as a function of A'''
        amin = self.Amin()
        A_ = np.linspace(amin, self.AM(0), 100)[:-1]  # remove Im=0 point
        p,q, I, F = self.p, self.q, self.I, self.F
        plt.title('Debt to equity ratio:  ' + r'$\frac{I+F-I^m}{I^m}$')
        Im = np.minimum(I + F, self.minmon(A_) * (q / (p - q)) * (1 / beta))
        de = np.divide(I + F - Im, Im, where=Im>0)
        plt.plot(A_, de)
        plt.xlabel('A -- pledgeable assets')
        plt.axvline(x=self.AM(self.mcross()), linestyle=':');
        plt.axhline(y=0, linestyle=':');
        plt.axvline(x=self.Amin(), linestyle=':')
        plt.xlim(amin-10, 140);
        plt.ylim(0, 20);

if __name__ == '__main__':
    """Sample use of the bankzone class """



