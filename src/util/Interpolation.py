import numpy as np


def _check_input(start_latent: np.ndarray, end_latent: np.ndarray, t: float) -> None:
    assert start_latent.shape == end_latent.shape
    assert len(start_latent.shape) == 1
    assert t >= 0 and t <= 1


def lerp(start_latent: np.ndarray, end_latent: np.ndarray, t: float) -> np.ndarray:
    """
    Return the linear interpolated latent code between two latent codes (`start_latent`,`end_latent`)
    with interpolation parameter `t`.

    Args:
        start_latent (np.ndarray): 1-dim array containing a latent code.
        end_latent (np.ndarray): 1-dim array containing a latent code.
        t (float): Step-size where 0>=`t`>=1

    Returns:
        np.ndarray: A latent code from the interpolation.
    """
    _check_input(start_latent, end_latent, t)
    return (1 - t) * start_latent + t * end_latent


def slerp(start_latent: np.ndarray, end_latent: np.ndarray, t: float) -> np.ndarray:
    """
    Return the spherical interpolated latent code between two latent codes
    (`start_latent`,`end_latent`) with interpolation parameter `t`.

    Args:
        start_latent (np.ndarray): 1-dim array containing a latent code.
        end_latent (np.ndarray): 1-dim array containing a latent code.
        t (float): Step-size where 0>=`t`>=1

    Returns:
        np.ndarray: A latent code from the interpolation.
    """
    # inspired by https://github.com/soumith/dcgan.torch slerp implementation
    _check_input(start_latent, end_latent, t)
    omega = np.arccos(
        np.clip(
            np.dot(
                start_latent / np.linalg.norm(start_latent),
                end_latent / np.linalg.norm(end_latent),
            ),
            -1,
            1,
        )
    )
    so = np.sin(omega)
    if so == 0:
        return (1.0 - t) * start_latent + t * end_latent  # L'Hopital's rule/LERP
    return (
        np.sin((1.0 - t) * omega) / so * start_latent
        + np.sin(t * omega) / so * end_latent
    )
