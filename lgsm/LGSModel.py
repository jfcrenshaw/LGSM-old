from jax._src.nn.functions import normalize
import numpy as np
import jax
import jax.numpy as jnp
import elegy
from typing import Sequence
import sncosmo
from sncosmo.constants import HC_ERG_AA
from .utils import mag_to_flambda


class StandardScaler(elegy.Module):
    """Standard Scaler that ensures input dimensions have mean zero and unit variance."""

    def __init__(self, input_mean: np.ndarray, input_std: np.ndarray, **kwargs):
        super().__init__()
        self.input_mean = input_mean
        self.input_std = input_std

    def call(self, inputs, **kwargs):
        return (inputs - self.input_mean) / self.input_std


class IdentityLayer:
    """Identity layer that stands in for BatchNormalization when batch_norm == False."""

    def __call__(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return x


class Encoder(elegy.Module):
    """Encoder that maps redshift, photometry, and photometric errors
    onto intrinsic and extrinsic latent variables.
    """

    def __init__(
        self,
        encoder_layers: Sequence[int],
        intrinsic_latent_size: int,
        batch_norm: bool,
        **kwargs,
    ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.intrinsic_latent_size = intrinsic_latent_size
        # hard-code size of extrinsic latent variables
        # this is because the user cannot change this because changing the
        # extrinsic latent variables requires you to change the PhysicsLayer
        self.extrinsic_latent_size = 1
        # setup the NormLayer
        if batch_norm:
            self.NormLayer = elegy.nn.BatchNormalization
        else:
            self.NormLayer = IdentityLayer

    def call(self, inputs: np.ndarray, **kwargs) -> dict:

        # pull out the redshift, which I will assume is the first column
        redshift = inputs[:, 0, None]

        # construct the first encoder layers
        for layer in self.encoder_layers:
            inputs = elegy.nn.Linear(layer)(inputs)
            inputs = self.NormLayer()(inputs)
            inputs = jax.nn.relu(inputs)
            self.add_summary("relu", jax.nn.relu, inputs)

        # calculate the intrinsic latent variables, which are from a normal dist.
        mean = elegy.nn.Linear(self.intrinsic_latent_size, name="linear_mean")(inputs)
        log_stds = elegy.nn.Linear(self.intrinsic_latent_size, name="linear_std")(
            inputs
        )
        stds = jnp.exp(log_stds)
        intrinsic_latents = mean + stds * jax.random.normal(self.next_key(), mean.shape)

        # calculate the extrinsic latent variables (not including redshift,
        # which was pulled out of the inputs above)
        extrinsic_latents = elegy.nn.Linear(
            self.extrinsic_latent_size, name="linear_extrinsic"
        )(inputs)
        amplitude = extrinsic_latents[:, 0, None]

        return {
            "intrinsic_latents": intrinsic_latents,
            "amplitude": amplitude,
            "redshift": redshift,
        }


class Decoder(elegy.Module):
    """Decoder that maps intrinsic latents onto an SED in AB magnitude."""

    def __init__(
        self,
        decoder_layers: Sequence[int],
        sed_bins: int,
        sed_unit: str,
        batch_norm: bool,
        **kwargs,
    ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.sed_bins = sed_bins
        self.sed_unit = sed_unit
        # setup the NormLayer
        if batch_norm:
            self.NormLayer = elegy.nn.BatchNormalization
        else:
            self.NormLayer = IdentityLayer

    def call(self, intrinsic_latents: np.ndarray, **kwargs) -> dict:

        # construct the first decoder layers
        for layer in self.decoder_layers:
            intrinsic_latents = elegy.nn.Linear(layer)(intrinsic_latents)
            intrinsic_latents = self.NormLayer()(intrinsic_latents)
            intrinsic_latents = jax.nn.relu(intrinsic_latents)
            self.add_summary("relu", jax.nn.relu, intrinsic_latents)

        # calculate the SED in AB magnitudes
        sed = elegy.nn.Linear(self.sed_bins)(intrinsic_latents)

        return {f"sed_{self.sed_unit}": sed}


class VAE(elegy.Module):
    """VAE that combines the Encoder and Decoder."""

    def __init__(
        self,
        # Encoder settings
        encoder_layers: Sequence[int],
        intrinsic_latent_size: int,
        # Decoder settings
        decoder_layers: Sequence[int],
        sed_bins: int,
        sed_unit: str,
        # global settings
        batch_norm: bool,
        **kwargs,
    ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.intrinsic_latent_size = intrinsic_latent_size
        self.decoder_layers = decoder_layers
        self.sed_bins = sed_bins
        self.sed_unit = sed_unit
        self.batch_norm = batch_norm

    def call(self, inputs: np.ndarray, **kwargs) -> dict:

        # encode the photometry in the implicit/explicit latent space
        latents = Encoder(
            self.encoder_layers, self.intrinsic_latent_size, self.batch_norm
        )(inputs)

        # decode into an SED
        sed = Decoder(
            self.decoder_layers, self.sed_bins, self.sed_unit, self.batch_norm
        )(**latents)

        return {**latents, **sed}


class PhysicsLayer(elegy.Module):
    """A physics layer that calculates photometry from an SED in AB magnitude.

    1. Calculating fluxes from F_lambda

    CCDs are photon counters, so we must convert from energy (ergs) to photon
    counts. We can do this via dividing by the energy/photon:
    energy/photon = hc / lambda.

    To calculate the flux through a bandpass, we convolve this with the
    dimensionless filter response function, T(lambda):
    flux = 1/hc * Integral[dlambda lambda T(lambda) F_lambda(lambda)].
    The units of this flux are photon/s/cm^2.

    2. Converting this flux back to an AB magnitude.

    We need to divide this flux by the flux of an object with magnitude zero
    in the corresponding band (i.e. the integrated flux of the 3631 Jy SED
    through the corresponding band). We call this the zero point (zp) flux.

    Thus, the AB magnitude through the band is m_AB = -2.5 * log10(flux/zp).
    """

    def __init__(
        self,
        sed_min: float,
        sed_max: float,
        sed_bins: int,
        sed_unit: str,
        bandpasses: Sequence[str],
        band_oversampling: int,
        **kwargs,
    ):
        super().__init__()
        self.sed_min = sed_min
        self.sed_max = sed_max
        self.sed_bins = sed_bins
        self.sed_unit = sed_unit
        self.bandpasses = bandpasses
        self.band_oversampling = band_oversampling

        # save the SED wavelength grid
        self.sed_dwave = (sed_max - sed_min) / (sed_bins - 1)
        self.sed_wave = np.arange(
            self.sed_min, self.sed_max + self.sed_dwave, self.sed_dwave
        )

        # setup functions to handle sed units
        if self.sed_unit == "mag":
            self.scale_sed = lambda sed, amplitude: amplitude + sed
            self.convert_sed_units = lambda sed: mag_to_flambda(sed, self.sed_wave)
        else:  # sed_unit == "flambda"
            self.scale_sed = lambda sed, amplitude: amplitude * sed
            self.convert_sed_units = lambda sed: sed

        # precompute the  weights for flux integration
        self._setup_band_weights()

    def _setup_band_weights(self):
        """Precompute the weights for flux integration so they can be reused
        over and over again!

        By our definition,
        integrated flux = (F_lambda * weights).sum().

        Note that you can optionally oversample the bandpasses. This is so they
        have a finer wavelength sampling than the SED itself. This can reduce
        computational errors in calculating the integrated flux.
        """

        # check if oversampling bands
        assert (
            self.band_oversampling % 2 == 1
        ), "band_oversampling must be an odd integer."
        pad = (self.band_oversampling - 1) // 2

        # wavelength grid for bandpass is essentially the same as sed_wave,
        # potentially with oversampling
        band_dwave = self.sed_dwave / self.band_oversampling
        band_wave = jnp.arange(
            self.sed_min - band_dwave * pad,
            self.sed_max + self.sed_dwave + band_dwave * pad,
            band_dwave,
        )

        # set the overall flux normalization
        norm = 1 / HC_ERG_AA  # photons / erg / AA

        # calculate the weights for each bandpass
        band_weights = []
        for band in self.bandpasses:

            # get the integrated zero point flux for the band
            zp = sncosmo.get_magsystem("ab").zpbandflux(band)

            # get the filter response function
            T = sncosmo.get_bandpass(band)(band_wave)

            # note we don't have to worry about normalizing the response
            # function T, because the same normalization appears in zp,
            # which cancels when we divide by zp.

            # calculate the weights needed for flux integration
            weights = norm / zp * T * band_wave * band_dwave

            # convolve the weights so that every point in conv_weights is
            # the sum of the adjacent N=oversampling elements in weights.
            conv_weights = jnp.convolve(
                weights, jnp.ones(self.band_oversampling), mode="valid"
            )
            band_weights.append(conv_weights)

        # save the band wavelength grid and the convolved weights
        self.band_wave = jnp.array(band_wave[pad : -pad or None])
        self.band_weights = jnp.array(band_weights)

    def _calc_mag_single_sed(self, sed_flambda, redshift, band_weights):

        # instead of redshifting the SED, we can blueshift the filters
        # (I'm not sure why, but doing so makes this simple calculation
        # far more accurate than redshifting SED - at least when comparing
        # to calculations with sncosmo...)
        band_weights = jnp.interp(
            self.sed_wave,
            self.band_wave / (1 + redshift),
            band_weights,
            left=0,
            right=0,
        )

        # calculate the integrated flux through the band
        flux = (sed_flambda * band_weights).sum()
        # convert to AB magnitude
        mag = -2.5 * jnp.log10(flux)

        return mag

    def _calc_mags_single_sed(self, sed_flambda, redshift):
        """Vectorize _calc_mag_single_sed to return mags for all bands."""
        return jax.vmap(
            lambda band_weights: self._calc_mag_single_sed(
                sed_flambda, redshift, band_weights
            )
        )(self.band_weights)

    def _calc_mags_multiple_seds(self, sed_flambda, redshift):
        """Vectorize _calc_mags_single_sed to return mags for multiple SEDs."""
        return jax.vmap(self._calc_mags_single_sed)(sed_flambda, redshift)

    def call(
        self, sed: np.ndarray, amplitude: np.ndarray, redshift: np.ndarray, **kwargs
    ) -> dict:
        """Calculate fluxes for the SEDs at the appropriate amplitude and redshifts."""

        sed_scaled = self.scale_sed(sed, amplitude)
        sed_flambda = self.convert_sed_units(sed_scaled)

        mags = self._calc_mags_multiple_seds(sed_flambda, redshift)

        return {"predicted_photometry": mags}


class LGSModel(elegy.Module):
    """The full Latent Galaxy SED Model (LGSM)."""

    def __init__(
        self,
        # StandardScaler settings
        input_mean: np.ndarray,
        input_std: np.ndarray,
        # Encoder settings
        encoder_layers: Sequence[int],
        intrinsic_latent_size: int,
        # Decoder settings
        decoder_layers: Sequence[int],
        sed_min: float,
        sed_max: float,
        sed_bins: int,
        sed_unit: str,
        # VAE settings
        batch_norm: bool,
        # PhysicsLayer settings
        bandpasses: Sequence[str],
        band_oversampling: int,
        **kwargs,
    ):
        super().__init__()
        self.input_mean = input_mean
        self.input_std = input_std
        self.encoder_layers = encoder_layers
        self.intrinsic_latent_size = intrinsic_latent_size
        self.decoder_layers = decoder_layers
        self.sed_min = sed_min
        self.sed_max = sed_max
        self.sed_bins = sed_bins
        self.sed_unit = sed_unit
        self.batch_norm = batch_norm
        self.bandpasses = bandpasses
        self.band_oversampling = band_oversampling

    def call(self, inputs: np.ndarray, **kwargs) -> dict:

        inputs = StandardScaler(self.input_mean, self.input_std)(inputs)

        vae_outputs = VAE(
            self.encoder_layers,
            self.intrinsic_latent_size,
            self.decoder_layers,
            self.sed_bins,
            self.sed_unit,
            self.batch_norm,
        )(inputs)

        predicted_photometry = PhysicsLayer(
            self.sed_min,
            self.sed_max,
            self.sed_bins,
            self.sed_unit,
            self.bandpasses,
            self.band_oversampling,
        )(sed=vae_outputs[f"sed_{self.sed_unit}"], **vae_outputs)

        return {**vae_outputs, **predicted_photometry}
