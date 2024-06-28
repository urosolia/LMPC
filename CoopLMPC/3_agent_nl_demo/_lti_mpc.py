import numpy as np
import scipy.linalg as sla

'''Internal functions for formulating MPC problem as QP'''

def _BuildMatCost(self):
    N = self.N
    Q = self.Q
    R = self.R
    P = self.P
    Q_slack = self.Q_slack
    b_slack = self.b_slack
    R_d = self.R_d

    x_ref = self.x_ref

    b = [Q] * (N)
    Mx = sla.block_diag(*b)

    c = [R + 2*R_d] * (N) # Need to add R_d for the derivative input cost
    Mu = sla.block_diag(*c)

    # Need to consider that the last input appears just once in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - np.diag(R_d)[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - np.diag(R_d)[0]

    # Derivative Input Cost
    OffDiaf = -np.tile(np.diag(R_d), N-1)
    np.fill_diagonal(Mu[2:], OffDiaf)
    np.fill_diagonal(Mu[:, 2:], OffDiaf)

    quadLaneSlack = Q_slack * np.eye(2*N)
    M00 = sla.block_diag(Mx, P, Mu)
    M0  = sla.block_diag(M00, quadLaneSlack)

    # xtrack = np.array([vt, 0, 0, 0, 0, 0])
    # q_hard = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M00)
    q_hard = - 2 * np.dot(np.append(x_ref.flatten(), np.zeros(R.shape[0] * N)), M00)

    linLaneSlack = b_slack * np.ones(2*N)
    q = np.append(q_hard, linLaneSlack)

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost
    # M = M0

    return M, q

def _BuildMatEqConst(self):
    # Build matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    n = self.n_x
    N = self.N
    d = self.n_u
    A, B = self.A, self.B

    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A
        Gu[np.ix_(ind1, ind2u)] = -B

    G_hard = np.hstack((Gx, Gu))
    SlackLane = np.zeros((G_hard.shape[0], 2*N))
    G = np.hstack((G_hard, SlackLane))

    return G, E

def _BuildMatIneqConst(self):
    N = self.N
    n = self.n_x
    a_l = self.a_lim[0]
    a_u = self.a_lim[1]
    df_l = self.df_lim[0]
    df_u = self.df_lim[1]

    track_width = self.track.track_width
    track_slack = self.track.slack

    # Build the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])

    # Track boundary limits are intentionally set large so we can add slack later
    bx = np.array([[track_width/2 + track_slack],    # max ey
                   [track_width/2 + track_slack]])  # min ey

    # Build the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[df_u],  # Max Steering
                   [-df_l],  # Min Steering
                   [a_u],  # Max Acceleration
                   [-a_l]])  # Min Acceleration

    # bu = np.array([[a_u],  # Max Steering
    #                [-a_l],  # Min Steering
    #                [df_u],  # Max Acceleration
    #                [-df_l]])  # Min Acceleration

    # Now stack the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    rep_a = [Fx] * (N)
    Mat = sla.block_diag(*rep_a)
    NoTerminalConstr = np.zeros((np.shape(Mat)[0], n))  # No need to constraint also the terminal point
    Fxtot = np.hstack((Mat, NoTerminalConstr))
    # Fxtot = sla.block_diag(*rep_a)
    bxtot = np.tile(np.squeeze(bx), N)

    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu] * (N)
    Futot = sla.block_diag(*rep_b)
    butot = np.tile(np.squeeze(bu), N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack((Fxtot, np.zeros((rFxtot, cFutot))))
    Dummy2 = np.hstack((np.zeros((rFutot, cFxtot)), Futot))
    F_hard = np.vstack((Dummy1, Dummy2))
    b = np.hstack((bxtot, butot))

    LaneSlack = np.zeros((F_hard.shape[0], 2*N))
    colIndexPositive = []
    rowIndexPositive = []
    colIndexNegative = []
    rowIndexNegative = []
    for i in range(0, N):
        colIndexPositive.append( i*2 + 0 )
        colIndexNegative.append( i*2 + 1 )

        rowIndexPositive.append(i*Fx.shape[0] + 0) # Slack on second element of Fx
        rowIndexNegative.append(i*Fx.shape[0] + 1) # Slack on third element of Fx

    LaneSlack[rowIndexPositive, colIndexPositive] =  1.0
    LaneSlack[rowIndexNegative, rowIndexNegative] = -1.0

    F = np.hstack((F_hard, LaneSlack))

    return F, b

def _BuildTermConstr(self, G, E, L, Eu):
    raise NotImplementedError('Terminal constraint not implemented')
