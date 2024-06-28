import numpy as np

from dynamics_models import CT_Kin_Bike_Model, DT_Kin_Bike_Model

class CT_Kin_Bike_Agent(CT_Kin_Bike_Model):
    def __init__(self, l_r, l_f, w, dt, x_0, x_f, df_lim, a_lim):
        super(CT_Kin_Bike_Agent, self).__init__(l_r, l_f)

        self.w = w
        self.l = l_r + l_f
        self.r = np.sqrt((self.w/2.0)**2 + (self.l/2.0)**2)
        self.dt = dt
        self.x_0 = x_0
        self.x_f = x_f

        self.state_his = [x_0]
        self.input_his = []

        self.a_lim = [-1.0, 1.0]
    	self.df_lim = [-0.5, 0.5]

        # Build the matrices for the input constraint
        self.F_u = np.array([[1., 0.],
                       [-1., 0.],
                       [0., 1.],
                       [0., -1.]])

        df_l, df_u = self.df_lim
        a_l, a_u = self.a_lim
        self.b_u = np.array([[df_u],  # Max Steering
                       [-df_l],  # Min Steering
                       [a_u],  # Max Acceleration
                       [-a_l]])  # Min Acceleration

    def get_DT_jacs(self, x, u, eps):
        A_c, B_c, c_c = self.get_numerical_jacs(x, u, eps)

        A_d = np.eye(self.n_x) + self.dt*A_c
        B_d = self.dt*B_c
        c_d = self.dt*c_c

        return A_d, B_d, c_d

    def update_state_input(self, x, u):
        self.state_his.append(x)
        self.input_his.append(u)

    def get_input_constraints(self):
        return self.F_u, self.b_u


class DT_Kin_Bike_Agent(DT_Kin_Bike_Model):
    def __init__(self, l_r, l_f, w, dt, x_0, x_f,
        a_lim=[-1.0, 1.0], df_lim=[-0.5, 0.5], x_lim=[-10.0, 10.0],
        y_lim=[-10.0, 10.0], psi_lim=None, v_lim=[-2.0, 2.0]):
        super(DT_Kin_Bike_Agent, self).__init__(l_r, l_f)

        self.w = w
        self.l = l_r + l_f
        self.r = np.sqrt((self.w/2.0)**2 + (self.l/2.0)**2)
        self.dt = dt
        self.x_0 = x_0
        self.x_f = x_f

        self.state_his = [x_0]
        self.input_his = []

        self.a_lim = a_lim
    	self.df_lim = df_lim

        self.x_lim = x_lim
        self.y_lim = y_lim
        self.psi_lim = psi_lim
        self.v_lim = v_lim

        # Build the matrices for the input constraint
        F = []
        b = []
        if self.x_lim is not None:
            F.append(np.array([[1., 0., 0., 0.], [-1., 0., 0., 0.]]))
            b.append(np.array([[x_lim[1]], [x_lim[0]]]))

        if self.y_lim is not None:
            F.append(np.array([[0., 1., 0., 0.], [0., -1., 0., 0.]]))
            b.append(np.array([[y_lim[1]], [y_lim[0]]]))

        if self.psi_lim is not None:
            F.append(np.array([[0., 0., 1., 0.], [0., 0., -1., 0.]]))
            b.append(np.array([[psi_lim[1]], [psi_lim[0]]]))

        if self.v_lim is not None:
            F.append(np.array([[0., 0., 0., 1.], [0., 0., 0., -1.]]))
            b.append(np.array([[v_lim[1]], [v_lim[0]]]))

        self.F = np.vstack(F)
        self.b = np.vstack(b)

        H = []
        g = []
        if self.df_lim is not None:
            H.append(np.array([[1., 0.], [-1., 0.]]))
            g.append(np.array([[df_lim[1]], [df_lim[0]]]))

        if self.a_lim is not None:
            H.append(np.array([[0., 1.], [0., -1.]]))
            g.append(np.array([[a_lim[1]], [a_lim[0]]]))

        self.H = np.vstack(H)
        self.g = np.vstack(g)

    def get_DT_jacs(self, x, u):
       A, B, c = self.get_numerical_jacs(x, u)

       return A, B, c

    def update_state_input(self, x, u):
        self.state_his.append(x)
        self.input_his.append(u)

    def get_state_constraints(self):
        return self.F, self.b

    def get_input_constraints(self):
        return self.H, self.g

    def get_collision_buff_r(self):
        return self.r
