def pack_unknowns(ln_r_edge, u):
    """
    Pack x = [ln_r_edge[1:-1], u].
    """
    return np.concatenate((ln_r_edge[1:-1], u))


def unpack_unknowns(x, r_outer, n_shell):
    """
    Unpack x into ln_r_edge and u.
    """
    nr = n_shell - 1

    ln_r_edge = np.empty(n_shell + 1, dtype=constants.FLOATDTYPE)

    ln_r_edge[0] = -np.inf
    ln_r_edge[-1] = np.log(r_outer)
    ln_r_edge[1:-1] = x[:nr]

    u = x[nr:]

    return ln_r_edge, u
