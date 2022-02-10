#与えられた行列AをLU分解して, Ax=bをO(N^2)で解く
import numpy as np
class LUdecomposition:
    def __init__(self,A):
        self.A=A.copy()
        self.n=len(A)
        self.p=np.arange(self.n)
        for k in range(self.n):
            idx, max_value=-1,-1.
            for i in range(k,self.n):
                if max_value<np.abs(self.A[i][k]):
                    idx=i
                    max_value=abs(self.A[i][k])
            self.p[k],self.p[idx]=self.p[idx],self.p[k]
            self.A[k],self.A[idx]=self.A[idx],self.A[k]
            w=1./self.A[k][k]
            for i in range(k+1,self.n):
                self.A[i][k]=self.A[i][k]*w
                for j in range(k+1,self.n):
                    self.A[i][j]-=self.A[i][k]*self.A[k][j]

    def solve(self,b):
        x=b.copy()
        for i in range(self.n): x[i]=b[self.p[i]]
        for k in range(self.n):
            for i in range(k+1,self.n):
                x[i]-=self.A[i][k]*x[k]
        for k in reversed(range(self.n)):
            for i in range(k+1,self.n):
                x[k]-=self.A[k][i]*x[i]
            x[k]/=self.A[k][k]
        return x
