from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete

class DynamicalSystem(ABC):
    @abstractmethod
    def dynamics(self, t, state, input_):
        pass
    
    @abstractmethod
    def cont2dis(self, dt):
        pass
    
class LTI(DynamicalSystem):
    def __init__(self, A, B, C=None, D=None, process_noise_cov=None, measurement_noise_cov=None):
        self.A = A
        self.B = B
        self.C = C if C is not None else np.eye(A.shape[0])
        self.D = D if D is not None else np.zeros((self.C.shape[0], B.shape[1]))
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        
    def dynamics(self, t, state, input_):
        return self.A @ state + self.B @ input_
    
    def noisy_dynamics(self, t, state, input_):
        noise = np.random.multivariate_normal(np.zeros(self.A.shape[0]), self.process_noise_cov) if self.process_noise_cov is not None else 0
        return self.dynamics(t, state, input) + noise
    
    def msmt(self, t, state, input_):
        return self.C @ state + self.D @ input_
    
    def noisy_msmt(self, t, state, input_):
        noise = np.random.multivariate_normal(np.zeros(self.C.shape[0]), self.measurement_noise_cov) if self.measurement_noise_cov is not None else 0
        return self.msmt(t, state, input_) + noise
    
    def cont2dis(self, dt):
        sys_d = cont2discrete(self.A, self.B, self.C, self.D)
        return sys_d[0], sys_d[1]
    
class LTV(DynamicalSystem):
    def __init__(self, A_func, B_func, C_func=None, D_func=None, process_noise_cov=None, measurement_noise_cov=None):
        self.A_func = A_func
        self.B_func = B_func
        self.C_func = C_func
        self.D_func = D_func
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov

    def dynamics(self, t, state, input_):
        A = self.A_func(t)
        B = self.B_func(t)
        return A @ state + B @ input_

    def noisy_dynamics(self, t, state, input_):
        A = self.A_func(t)
        B = self.B_func(t)
        noise = np.random.multivariate_normal(np.zeros(A.shape[0]), self.process_noise_cov) if self.process_noise_cov is not None else 0
        return A @ state + B @ input_ + noise

    def msmt(self, t, state, input_):
        if self.C_func and self.D_func:
            C = self.C_func(t)
            D = self.D_func(t)
        else:
            C = np.eye(len(state))
            D = np.zeros((len(state), len(input_)))
        return C @ state + D @ input_

    def noisy_msmt(self, t, state, input_):
        C = self.C_func(t) if self.C_func else np.eye(len(state))
        noise = np.random.multivariate_normal(np.zeros(C.shape[0]), self.measurement_noise_cov) if self.measurement_noise_cov is not None else 0
        return self.msmt(t, state, input_) + noise

    def cont2dis(self, dt):
        raise NotImplementedError("Discretization for LTV systems requires time-specific methods.")

class NonlinearSystem(DynamicalSystem):
    def __init__(self, process_noise_cov=None, measurement_noise_cov=None):
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov

    @abstractmethod
    def f(self, x, u):
        pass

    @abstractmethod
    def g(self, x, u):
        pass

    def dynamics(self, t, state, input_):
        return self.f(state, input_)

    def noisy_dynamics(self, t, state, input_):
        base_dynamics = self.dynamics(t, state, input_)
        noise = np.random.multivariate_normal(np.zeros(len(state)), self.process_noise_cov) if self.process_noise_cov is not None else 0
        return np.array(base_dynamics) + noise

    def msmt(self, t, state, input_):
        return self.g(state, input_)

    def noisy_msmt(self, t, state, input_):
        output = self.msmt(t, state, input_)
        noise = np.random.multivariate_normal(np.zeros(len(output)), self.measurement_noise_cov) if self.measurement_noise_cov is not None else 0
        return np.array(output) + noise

    def continuous_to_discrete(self, dt):
        # TODO
        return 0

class GlucoseInsulinModel(NonlinearSystem):
    def __init__(self, params, process_noise_cov=None, measurement_noise_cov=None):
        super().__init__(process_noise_cov, measurement_noise_cov)
        self.params = params

    def f(self, x, u):
        Gp, Gt, Ip, Il, Qsto1, Qsto2, Qgut, XL, X = x
        D, delta, Fcns, X_input = u
        p = self.params

        Qsto = Qsto1 + Qsto2

        kempt = p['kmin'] + (p['kmax'] - p['kmin']) * \
            (np.tanh(p['alpha'] * Qsto - p['alpha'] * p['b'] * D) -
             np.tanh(p['beta'] * Qsto - p['beta'] * p['c'] * D) + 2) / 2

        EGP = p['kp1'] - p['kp2'] * Gp - p['kp3'] * XL
        Uii = Fcns
        Uid = ((p['Vm0'] + p['Vmx'] * X_input) * Gt) / (p['Km0'] + Gt)
        Ra = p['f'] * p['kabs'] * Qgut / p['BW']

        dGp = EGP + Ra - Uii - p['k1'] * Gp + p['k2'] * Gt
        dGt = -Uid + p['k1'] * Gp - p['k2'] * Gt
        dIp = -(p['m2'] + p['m4']) * Ip + p['m1'] * Il + p['IIR']
        dIl = -(p['m1'] + p['m3']) * Il + p['m2'] * Ip
        dQsto1 = -p['kgri'] * Qsto1 + D * delta
        dQsto2 = -kempt * Qsto2 + p['kgri'] * Qsto1
        dQgut = -p['kabs'] * Qgut + kempt * Qsto2
        dXL = -p['ki'] * (XL - Ip)
        dX = -p['p2U'] * X + p['p2U'] * Ip

        return [dGp, dGt, dIp, dIl, dQsto1, dQsto2, dQgut, dXL, dX]

    def g(self, x, u):
        # TODO
        return x  