from qkal.config import QKALReconstructionConfig
from qkal.train import train_qkal_from_arrays
from qkal.density import density_from_qkal
from qkal.plot import plot_y_space, plot_u_space

def load_your_data(x,y):
    #todo
    return x,y

if __name__ == "__main__":
    config = QKALReconstructionConfig()
    X, y = load_your_data()

    model, qt = train_qkal_from_arrays(X, y, config)

    y_of_u, f_y_batch, u_grid, rho_u_batch, dy_du, p_marg_y = density_from_qkal(
        model, X, y, config
    )

    plot_u_space(
        u_grid = u_grid,
        rho_u_batch = rho_u_batch,
        n_show=config.n_of_reconstructions
    )

    plot_y_space(
        y_of_u = y_of_u,
        f_y_batch = f_y_batch,
        y_data = y,
        p_marg_y = p_marg_y,
        n_show=config.n_of_reconstructions
    )
