import numpy as np
import matplotlib.pyplot as plt
from . import utils
from numbers import Number


def _plot_sed(
    wave: np.ndarray,
    sed: np.ndarray,
    z: float = 0,
    normalize_at: float = None,
    sed_unit: str = "mag",
    plot_unit: str = None,
    fig_settings: dict = {},
    ax_settings: dict = {},
    plot_settings: dict = {},
    ax=None,
):

    # check that redshift is valid
    if z < 0:
        raise ValueError("z must be non-negative.")
    # check that normalize at is valid
    if normalize_at is not None:
        if not isinstance(normalize_at, Number):
            raise ValueError("normalize_at must either be None or a number.")
        if normalize_at < wave.min() or normalize_at > wave.max():
            raise ValueError("normalize_at must be within range of wave.")
    # check that sed_unit and plot_unit are valid
    units_allowed = ["mag", "flambda", "fnu"]
    if sed_unit not in units_allowed[:2]:
        raise ValueError(f"sed_unit must be one of {', '.join(units_allowed[:2])}.")
    if plot_unit is None:
        plot_unit = sed_unit
    if plot_unit not in units_allowed:
        raise ValueError(f"plot_unit must be one of {', '.join(units_allowed)}.")

    # redshift the sed
    wave = wave * (1 + z)
    if sed_unit == "mag":
        sed = sed - 2.5 * np.log10(1 + z)
    else:
        sed = sed / (1 + z)

    # alias wave because it might be converted to frequency insead
    x = wave
    if sed_unit != plot_unit:
        sed = getattr(utils, f"{sed_unit}_to_{plot_unit}")(sed, wave=x)
    if plot_unit == "fnu":
        x = utils.wave_to_freq(x)[::-1]

    # normalize the SED at the requested location
    if normalize_at is not None:
        idx = np.argmin(np.abs(x - normalize_at))
        if plot_unit == "mag":
            sed = sed - sed[idx]
        else:
            sed = sed / sed[idx]

    # if ax not provided, we will make a new plot
    if ax is None:
        new_ax = True
    else:
        new_ax = False

    if new_ax:
        # if no fig_settings supplied, use defaults
        if not fig_settings:
            fig_settings = {
                "figsize": (3.5, 3.5),
                "dpi": 120,
                "constrained_layout": True,
            }
        # create the figure
        fig, ax = plt.subplots(**fig_settings)
        # if we are plotting mags, invert the y-axis
        if plot_unit == "mag":
            ax.invert_yaxis()

    # plot the sed
    ax.plot(x, sed, **plot_settings)

    # set the axis labels depending on plot_unit
    if plot_unit == "mag":
        xlabel = "Wavelength (AA)"
        ylabel = "AB Magnitude"
    elif plot_unit == "flambda":
        xlabel = "Wavelength (AA)"
        ylabel = "Flux Density (erg/s/AA/cm^2)"
    else:
        xlabel = "Frequency (Hz)"
        ylabel = "Flux Density (erg/s/Hz/cm^2)"
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # set the user-supplied axis settings
    ax.set(**ax_settings)

    # if we made a new figure and axis, return them!
    if new_ax:
        return fig, ax
    # otherwise return nothing
    else:
        return None, None


def plot_sed(
    wave: np.ndarray,
    sed: np.ndarray,
    z: np.ndarray = np.zeros(1),
    normalize_at: float = None,
    sed_unit: str = "mag",
    plot_unit: str = None,
    fig_settings: dict = {},
    ax_settings: dict = {},
    plot_settings: dict = {},
    ax=None,
):

    # save the settings that all function calls receive
    settings = {
        "normalize_at": normalize_at,
        "sed_unit": sed_unit,
        "plot_unit": plot_unit,
        "fig_settings": fig_settings,
        "ax_settings": ax_settings,
    }

    # enforce correct shapes
    wave = np.atleast_2d(wave)
    sed = np.atleast_2d(sed)
    z = np.reshape(z, (-1, 1))

    # if multiple wavelength grid, seds, or redshifts we provided
    # then we need to plot in a loop over them
    if wave.shape[0] > 1 or sed.shape[0] > 1 or z.size > 1:

        # broadcast wave and sed to a common shape
        wave, sed = np.broadcast_arrays(wave, sed)

        # if only one redshift, broadcast to shape for wave, sed
        if z.size == 1:
            z = np.broadcast_to(z, (wave.shape[0], 1))
        # if multiple redshifts, plot each sed at each redshift
        else:
            wave, sed, z = (
                np.tile(wave, z.size).reshape(-1, wave.shape[1]),
                np.tile(sed, z.size).reshape(-1, sed.shape[1]),
                np.tile(z.flatten(), wave.shape[0]).reshape(-1, 1),
            )

        # broadcast plot_settings too
        for key, vals in plot_settings.items():
            vals = np.array([vals]).flatten()
            vals = np.tile(vals, np.ceil(z.size / vals.size).astype("int"))
            plot_settings[key] = vals

        # if no ax supplied, the first call will create an ax, which we
        # will pass to all subsequent calls. We will also return the new
        # fig and ax
        if ax is None:
            fig, ax = _plot_sed(
                wave=wave[0],
                sed=sed[0],
                z=z[0],
                plot_settings={
                    key: plot_settings[key][0] for key in plot_settings.keys()
                },
                **settings,
            )
            for i, (W, S, Z) in enumerate(zip(wave[1:], sed[1:], z[1:])):
                _plot_sed(
                    wave=W,
                    sed=S,
                    z=Z,
                    ax=ax,
                    plot_settings={
                        key: plot_settings[key][i + 1] for key in plot_settings.keys()
                    },
                    **settings,
                )
            return fig, ax
        # if an ax was supplied, we can do a simple loop and provide it every time
        else:
            for i, (W, S, Z) in enumerate(zip(wave, sed, z)):
                _plot_sed(
                    wave=W,
                    sed=S,
                    z=Z,
                    ax=ax,
                    plot_settings={
                        key: plot_settings[key][i] for key in plot_settings.keys()
                    },
                    **settings,
                )
            return None, None

    # otherwise, just plot the single set given
    else:

        return _plot_sed(
            wave=wave[0],
            sed=sed[0],
            z=z[0],
            ax=ax,
            plot_settings=plot_settings,
            **settings,
        )
