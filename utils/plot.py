import matplotlib.pyplot as plt

def plot_u_space(u_grid, rho_u_batch, n_show=10):
    """
    Rysuje rho(u|x) w u-space, z referencją = 1
    """
    plt.figure(figsize=(8,4))
    for i in range(min(n_show, rho_u_batch.shape[0])):
        plt.plot(u_grid.cpu(), rho_u_batch[i].cpu(), label=f"rho(u|x[{i}])")
    plt.axhline(1.0, color="C0", linestyle="--", label="uniform ref")
    plt.xlabel("u ∈ (0,1)")
    plt.ylabel("ρ(u|x)")
    plt.title("Model density in u-space ρ(u|x)")
    plt.legend(fontsize=8)
    plt.show()


def plot_y_space(y_of_u, f_y_batch, y_data, p_marg_y, n_show=10):
    """
    Rysuje f(y|x) w y-space, z referencją = KDE(y)
    """
    plt.figure(figsize=(8,4))
    for i in range(min(n_show, f_y_batch.shape[0])):
        plt.plot(y_of_u.cpu(), f_y_batch[i].cpu(), label=f"f(y|x[{i}])")
    plt.plot(y_of_u.cpu(), p_marg_y.cpu(), "k--", lw=2, label="marginal KDE")
    plt.xlabel("y")
    plt.ylabel("density")
    plt.title("Conditional density in y-space f(y|x)")
    plt.legend(fontsize=8)
    plt.show()
